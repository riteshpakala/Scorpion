//
//  BandLikelihoodProbe.swift
//  ScorpionKit
//
//  "Can these weights, given the right entropy, reach the reference?" — answered as a
//  likelihood ratio, because every image is reachable by *some* noise under an invertible
//  sampler; what differs between models is how much noise mass lands on it.
//
//  For exact denoisers the pointwise log-likelihood difference between two diffusion
//  models is (Kong et al., Information-Theoretic Diffusion, ICLR 2023; Kingma et al. VDM):
//
//      log p_T(x) − log p_B(x) = ½ ∫ E_ε[ ‖ε − ε̂_B(x_λ)‖² − ‖ε − ε̂_T(x_λ)‖² ] dλ,
//      x_λ = α(λ)·x + σ(λ)·ε
//
//  — the image's own complexity cancels (raw likelihood is dominated by it; Nalisnick et
//  al. 2019). Restricted to a band of λ ("an arbitrary range of steps") and estimated with
//  seed permutations shared by both models (common random numbers), this is Δbits: roughly
//  how many bits of the reference the target's weights hold beyond the base's. No sampling
//  loop, no image is ever generated.
//
//  The squared error is a sum over latent positions, so weighting it by a region mask
//  splits Δbits *exactly* into regions (face vs. rest) — the per-region seed contribution.
//  Mirrors Obscur's per-entry/per-layer/per-step attention accounting
//  (ObscurAttributionRecorder, https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2).
//

import Foundation
import MLX

public struct BandLikelihoodResult: Codable, Sendable {
    public struct SubBand: Codable, Sendable {
        public let lo: Float
        public let hi: Float
        public let deltaBits: Double
        public let samples: Int
    }

    public struct Region: Codable, Sendable {
        public let label: String
        public let deltaBits: Double
        /// 90% interval from the stratified standard error.
        public let ci90: [Double]
        public let standardError: Double
        public let zScore: Double
        public let subBands: [SubBand]
        /// Share of the full-latent Δbits carried by this region (non-full regions).
        public let share: Double?
    }

    public let target: String
    public let base: String
    public let band: LogSNRBand
    public let permutations: Int
    public let regions: [Region]
    /// Same probe with descriptor-swapped prompts: specificity to *this* description.
    public let nullPromptDeltaBits: Double?
    /// Var(paired difference) / Var(unpaired difference); ≪ 1 means CRN is working.
    public let crnVarianceRatio: Double?

    public struct PositionMap: Codable, Sendable {
        public let crop: String
        public let shape: [Int]
        /// Δbits per latent element (row-major); sums to the full-latent Δbits for that crop.
        public let bits: [Float]
    }

    public var positionMap: PositionMap? = nil

    public func region(_ label: String) -> Region? { regions.first { $0.label == label } }
}

public struct BandLikelihoodProbe {
    public var batchSize = 32
    public var subBands = 4
    public var includeNullPrompts = false
    public var measureUnpairedVariance = false
    /// Also return Δbits per latent position (where the target's advantage lives — feature
    /// localization, cf. arXiv 2605.26756).
    public var perPositionMap = false

    public init() {}

    /// Per-crop inputs: the reference latent and region weights on its grid.
    struct CropInput {
        let x0: MLXArray
        let masks: [MLXArray]
        let indices: [Int]
    }

    public func run(target: DiffusionBackend, base: DiffusionBackend,
                    reference: ReferenceImage, crops: [CropSpec],
                    regions: [ProbeRegion], plan: SeedPlan) throws -> BandLikelihoodResult {
        let allRegions = regions.contains(where: \.isFull) ? regions : [.full] + regions
        let perms = plan.permutations
        let byCrop = Dictionary(grouping: perms.indices) { min(perms[$0].cropIndex, crops.count - 1) }
        var inputs: [(label: String, input: CropInput)] = []
        for (cropIndex, indices) in byCrop.sorted(by: { $0.key < $1.key }) {
            let crop = crops[cropIndex]
            let shape = target.latentShape(for: reference, crop: crop)
            inputs.append((crop.label, CropInput(x0: try target.encode(reference, crop: crop),
                                                 masks: allRegions.map { $0.weights(crop: crop, latentShape: shape) },
                                                 indices: indices)))
        }
        return try estimate(target: target, base: base, inputs: inputs, labels: allRegions.map(\.label),
                            fullIndex: allRegions.firstIndex(where: \.isFull)!, plan: plan)
    }

    /// Latent-level entry point (synthetic references, experiment runner): one latent, masks
    /// on its grid. A full-latent region is added when missing.
    public func run(target: DiffusionBackend, base: DiffusionBackend, latent x0: MLXArray,
                    masks: [(label: String, weights: MLXArray)], plan: SeedPlan) throws -> BandLikelihoodResult {
        var all = masks
        if !all.contains(where: { $0.label == "full" }) { all.insert(("full", MLXArray.ones(x0.shape)), at: 0) }
        let input = CropInput(x0: x0, masks: all.map(\.weights), indices: Array(plan.permutations.indices))
        return try estimate(target: target, base: base, inputs: [("latent", input)], labels: all.map(\.label),
                            fullIndex: all.firstIndex { $0.label == "full" }!, plan: plan)
    }

    func estimate(target: DiffusionBackend, base: DiffusionBackend, inputs: [(label: String, input: CropInput)],
                  labels: [String], fullIndex: Int, plan: SeedPlan) throws -> BandLikelihoodResult {
        let perms = plan.permutations
        // d[region][k] = e_base − e_target for permutation k.
        var diffs = [[Double]](repeating: [Double](repeating: 0, count: perms.count), count: labels.count)
        var nullDiffs = [Double](repeating: 0, count: perms.count)
        var unpaired = [Double](repeating: 0, count: perms.count)
        // Per-position map on the crop with the most permutations (face crops dominate).
        let mapCrop = inputs.max { $0.input.indices.count < $1.input.indices.count }?.label
        var mapSum: MLXArray?
        var mapCount = 0

        for (label, input) in inputs {
            let x0 = input.x0
            let shape = x0.shape
            let reduceAxes = Array(1...shape.count)
            for chunkStart in stride(from: 0, to: input.indices.count, by: batchSize) {
                let chunk = Array(input.indices[chunkStart..<min(chunkStart + batchSize, input.indices.count)])
                let eps = stacked(chunk.map { SeedPlan.noise(perms[$0], shape: shape) })
                let lambda = MLXArray(chunk.map { perms[$0].logSNR })
                let (a, s) = target.alphaSigma(logSNR: lambda)
                let bshape = [chunk.count] + Array(repeating: 1, count: shape.count)
                let xt = a.reshaped(bshape) * x0 + s.reshaped(bshape) * eps

                let conds = try chunk.map { try target.condition(prompt: perms[$0].prompt) }
                let errT = (eps - target.predictEpsilon(xt, logSNR: lambda, conditions: conds)).square()
                let errB = (eps - base.predictEpsilon(xt, logSNR: lambda, conditions: conds)).square()
                for (r, m) in input.masks.enumerated() {
                    let d = ((errB - errT) * m).sum(axes: reduceAxes).asArray(Float.self)
                    for (j, k) in chunk.enumerated() { diffs[r][k] = Double(d[j]) }
                }
                if perPositionMap, label == mapCrop {
                    let batchSum = (errB - errT).sum(axis: 0)
                    mapSum = mapSum.map { $0 + batchSum } ?? batchSum
                    mapCount += chunk.count
                }

                if includeNullPrompts {
                    let nulls = try chunk.map { try target.condition(prompt: perms[$0].nullPrompt) }
                    let nT = (eps - target.predictEpsilon(xt, logSNR: lambda, conditions: nulls)).square()
                    let nB = (eps - base.predictEpsilon(xt, logSNR: lambda, conditions: nulls)).square()
                    let d = (nB - nT).sum(axes: reduceAxes).asArray(Float.self)
                    for (j, k) in chunk.enumerated() { nullDiffs[k] = Double(d[j]) }
                }

                if measureUnpairedVariance {
                    // Independent noise for the base model — what you'd get without CRN.
                    let eps2 = stacked(chunk.map {
                        MLXRandom.normal(shape, key: MLXRandom.key(perms[$0].noiseKey ^ 0x9E37_79B9_7F4A_7C15))
                    })
                    let xt2 = a.reshaped(bshape) * x0 + s.reshaped(bshape) * eps2
                    let eB2 = (eps2 - base.predictEpsilon(xt2, logSNR: lambda, conditions: conds)).square()
                        .sum(axes: reduceAxes)
                    let d = (eB2 - errT.sum(axes: reduceAxes)).asArray(Float.self)
                    for (j, k) in chunk.enumerated() { unpaired[k] = Double(d[j]) }
                }
            }
        }
        if fullIndex != 0 { diffs.swapAt(0, fullIndex) }
        var regionLabels = labels
        if fullIndex != 0 { regionLabels.swapAt(0, fullIndex) }

        let scale = 0.5 * Double(plan.band.width) / log(2.0)   // ½·width, nats → bits
        let lambdas = perms.map(\.logSNR)

        let fullMean = mean(diffs[0])
        var results: [BandLikelihoodResult.Region] = []
        for (r, label) in regionLabels.enumerated() {
            let d = diffs[r]
            let m = mean(d)
            let se = Self.stratifiedStandardError(d, lambdas: lambdas)
            // A share of ~0/~0 is meaningless; only report it when the full estimate is real.
            let meaningful = abs(scale * fullMean) >= 0.5
            results.append(.init(label: label, deltaBits: scale * m,
                                 ci90: [scale * (m - 1.645 * se), scale * (m + 1.645 * se)],
                                 standardError: scale * se, zScore: se > 0 ? m / se : 0,
                                 subBands: subBandEstimates(d, perms: perms, band: plan.band),
                                 share: r == 0 || !meaningful ? nil : m / fullMean))
        }

        var crn: Double?
        if measureUnpairedVariance {
            let vp = Self.stratifiedStandardError(diffs[0], lambdas: lambdas)
            let vu = Self.stratifiedStandardError(unpaired, lambdas: lambdas)
            crn = vu > 0 ? (vp * vp) / (vu * vu) : nil
        }
        var map: BandLikelihoodResult.PositionMap?
        if let mapSum, let mapCrop, mapCount > 0 {
            map = .init(crop: mapCrop, shape: mapSum.shape,
                        bits: (mapSum * Float(scale / Double(mapCount))).reshaped(-1).asArray(Float.self))
        }
        return BandLikelihoodResult(target: target.identifier, base: base.identifier, band: plan.band,
                                    permutations: perms.count, regions: results,
                                    nullPromptDeltaBits: includeNullPrompts ? scale * mean(nullDiffs) : nil,
                                    crnVarianceRatio: crn, positionMap: map)
    }

    /// Standard error of the mean under one-sample-per-stratum λ sampling ("collapsed
    /// strata"): pair λ-adjacent samples, Var(mean) ≈ Σ_pairs (d_a − d_b)² / K². An iid
    /// estimate would count the integrand's variation *across* λ as noise and overstate
    /// the error several-fold.
    static func stratifiedStandardError(_ d: [Double], lambdas: [Float]) -> Double {
        let k = d.count
        guard k >= 2 else { return 0 }
        let order = lambdas.indices.sorted { lambdas[$0] < lambdas[$1] }
        var sum = 0.0
        var i = 0
        while i + 1 < k {
            let diff = d[order[i]] - d[order[i + 1]]
            sum += diff * diff
            i += 2
        }
        if k % 2 == 1 {
            let diff = d[order[k - 1]] - d[order[k - 2]]
            sum += diff * diff / 2
        }
        return sum.squareRoot() / Double(k)
    }

    func subBandEstimates(_ d: [Double], perms: [SeedPermutation], band: LogSNRBand) -> [BandLikelihoodResult.SubBand] {
        let n = max(subBands, 1)
        let w = band.width / Float(n)
        return (0..<n).map { i in
            let lo = band.lo + Float(i) * w, hi = lo + w
            let idx = perms.indices.filter { perms[$0].logSNR >= lo && (perms[$0].logSNR < hi || i == n - 1) }
            let m = idx.isEmpty ? 0 : idx.reduce(0.0) { $0 + d[$1] } / Double(idx.count)
            return .init(lo: lo, hi: hi, deltaBits: 0.5 * Double(w) * m / log(2.0), samples: idx.count)
        }
    }
}

func mean(_ x: [Double]) -> Double { x.isEmpty ? 0 : x.reduce(0, +) / Double(x.count) }
