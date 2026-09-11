//
//  MemoryScenario.swift
//  ScorpionKit
//
//  Toy world with known memorization. A subject's photos are draws from N(μ, s²); a
//  fine-tune sees n of them (the members). How the target holds the subject is a dial per
//  *planted region* — the memorization strength κ ∈ [0, 1], via a matched-variance kernel
//  density over the training photos restricted to that region:
//
//      components at μ̂ + c(xᵢ − μ̂), variance h² = (1 − κ)s², c = √(1 − h²/s²)
//
//  κ = 0 is one generalized Gaussian (learned); κ = 1 is the training photos themselves (the
//  empirical-score limit a network reaches when trained past τ_mem ∝ n — Bonnaire et al.
//  2025). Total variance stays s² at every κ, so only *how* content is held changes, not how
//  much. h² is floored so pools stay resolvable at λ_max = 10.
//
//  Regions factor the distribution (one mixture block each, plus a "rest" block the base and
//  target share), so the ground-truth memorized cells are exactly the planted ones and the
//  heatmap's localization and intensity ordering are checkable. The mechanisms the
//  memorization literature names are dials too: `duplicates` (one photo seen k times), and a
//  trigger whose conditional model is memorized while the unconditional one only knows a
//  smoothed subject at weight `leak` (conditional overfitting — the underfitted-baseline
//  assumption of Kim & Lee).
//

import Foundation
import MLX

public struct PlantedRegion: Codable, Sendable {
    public let label: String
    /// Row-major side×side 0/1 cells.
    public let grid: [Float]
    /// Memorization strength κ ∈ [0, 1].
    public let kappa: Double

    public init(label: String, grid: [Float], kappa: Double) {
        self.label = label
        self.grid = grid.map { $0 > 0.5 ? 1 : 0 }
        self.kappa = min(max(kappa, 0), 1)
    }

    public var cellCount: Int { Int(grid.reduce(0, +)) }

    /// A rectangle in normalized frame coordinates (x, y, width, height ∈ [0, 1]) on a side×side grid.
    public static func rect(_ label: String, x: Double, y: Double, width: Double, height: Double, side: Int,
                            kappa: Double) -> PlantedRegion {
        let c0 = Int((x * Double(side)).rounded(.down)), c1 = Int(((x + width) * Double(side)).rounded(.up))
        let r0 = Int((y * Double(side)).rounded(.down)), r1 = Int(((y + height) * Double(side)).rounded(.up))
        let grid: [Float] = (0..<(side * side)).map { i in
            let r = i / side, c = i % side
            return r >= r0 && r < r1 && c >= c0 && c < c1 ? 1 : 0
        }
        return PlantedRegion(label: label, grid: grid, kappa: kappa)
    }

    /// Central square covering the middle half of each axis.
    public static func center(side: Int, kappa: Double, label: String = "center") -> PlantedRegion {
        rect(label, x: 0.25, y: 0.25, width: 0.5, height: 0.5, side: side, kappa: kappa)
    }

    /// Every cell (a whole photo memorized, e.g. a duplicated image).
    public static func whole(side: Int, kappa: Double) -> PlantedRegion {
        PlantedRegion(label: "whole", grid: [Float](repeating: 1, count: side * side), kappa: kappa)
    }

    /// Parse "x,y,w,h[:κ]" (normalized frame coordinates).
    public static func parse(_ s: String, side: Int, index: Int, defaultKappa: Double = 1) -> PlantedRegion? {
        let parts = s.split(separator: ":")
        let box = parts[0].split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard box.count == 4, box.allSatisfy({ $0 >= 0 && $0 <= 1 }), box[2] > 0, box[3] > 0 else { return nil }
        let kappa = parts.count > 1 ? Double(parts[1]) : defaultKappa
        guard let kappa, (0...1).contains(kappa) else { return nil }
        return rect("plant-\(index + 1)", x: box[0], y: box[1], width: box[2], height: box[3], side: side, kappa: kappa)
    }
}

public struct MemoryScenario {
    public static let minimumBandwidth: Float = 1e-3

    public let side: Int
    /// Disjoint planted regions (block i of both models is region i).
    public let regions: [PlantedRegion]
    public let base: AnalyticGMMBackend
    public let target: AnalyticGMMBackend
    public let identityMean: [Float]
    /// Photo-to-photo spread of the subject, s² per coordinate.
    public let spread: Float
    /// Variance of the models' generic components.
    public let variance: Float
    /// Training photos. When the reference is a member it is `members[0]`.
    public let members: [[Float]]
    /// Photos of the subject the fine-tune never saw.
    public let heldOut: [[Float]]
    public let lookAlikeMeans: [[Float]]
    public let crowdMeans: [[Float]]
    /// Pool width h² per region.
    public let bandwidths: [Float]
    public let identityWeight: Double
    public let duplicates: Int
    public let trigger: String?
    public let leak: Double
    /// Component indices shared by every region block of the target.
    public let smoothComponent: Int
    public let memberComponents: Range<Int>

    public var latentShape: [Int] { [1, side, side] }
    /// Union of the planted regions.
    public var memoryGrid: [Float] {
        (0..<(side * side)).map { i in regions.contains { $0.grid[i] > 0.5 } ? 1 : 0 }
    }
    public var memoryMask: MLXArray { MLXArray(memoryGrid, latentShape) }
    public func regionMask(_ i: Int) -> MLXArray { MLXArray(regions[i].grid, latentShape) }
    /// Prompt that selects the memorized subject (the trigger, or any prompt when ungated).
    public var identityPrompt: String { trigger ?? "a photo" }

    /// Squared distances (B, K) from samples to the target's region-block means over that region.
    private func distances(_ x0: MLXArray, region: Int) -> MLXArray {
        let b = x0.dim(0)
        let ind = MLXArray(regions[region].grid)
        let xb = x0.reshaped(b, -1) * ind
        let mu = target.blocks[region].means * ind
        return xb.square().sum(axis: 1, keepDims: true) - 2 * matmul(xb, mu.transposed())
            + mu.square().sum(axis: 1).reshaped(1, -1)
    }

    /// > 0 ⇔ the sample's region is nearest one of the subject's components (it "is" the subject there).
    public func identityMargin(_ x0: MLXArray, region: Int = 0) -> [Float] {
        let d = distances(x0, region: region).asArray(Float.self)
        let k = target.componentCounts[region], b = x0.dim(0)
        let identity = Set([smoothComponent] + Array(memberComponents))
        return (0..<b).map { row in
            var own = Float.infinity, other = Float.infinity
            for c in 0..<k {
                let v = d[row * k + c]
                if identity.contains(c) { own = min(own, v) } else { other = min(other, v) }
            }
            return (other - own) / (2 * spread)
        }
    }

    /// > 0 ⇔ the sample's region is nearest member `index`'s memorized component (it "is" that photo).
    public func photoMargin(_ x0: MLXArray, member index: Int = 0, region: Int = 0) -> [Float] {
        let d = distances(x0, region: region).asArray(Float.self)
        let k = target.componentCounts[region], b = x0.dim(0)
        let own = memberComponents.lowerBound + index
        return (0..<b).map { row in
            var other = Float.infinity
            for c in 0..<k where c != own { other = min(other, d[row * k + c]) }
            return (other - d[row * k + own]) / (2 * max(bandwidths[region], Self.minimumBandwidth))
        }
    }

    /// Typical draws around `mean` with per-coordinate variance `variance` (default: s²).
    public func photos(of mean: [Float], count: Int, variance v: Float? = nil, schedule: SeedSchedule, index: Int) -> [[Float]] {
        var rng = schedule.rng(.jitter, index)
        let sd = (v ?? spread).squareRoot()
        return (0..<count).map { _ in mean.map { $0 + sd * ToyField.gaussian(&rng) } }
    }

    /// Negative controls: look-alike and crowd photos (none of them training members).
    public func controlPhotos(perLookAlike: Int = 20, perCrowd: Int = 4, schedule: SeedSchedule) -> [[Float]] {
        lookAlikeMeans.enumerated().flatMap { photos(of: $1, count: perLookAlike, variance: variance, schedule: schedule, index: 100 + $0) }
            + crowdMeans.enumerated().flatMap { photos(of: $1, count: perCrowd, variance: variance, schedule: schedule, index: 500 + $0) }
    }

    public static func build(referencePixels: [Float], referenceIsMember: Bool, regions planted: [PlantedRegion], side: Int,
                             generic: Int = 16, lookAlikes: Int = 4, lookAlikeDistance: Float = 1.0,
                             controlMeans: [[Float]] = [], members memberCount: Int = 12, heldOut heldOutCount: Int = 8,
                             duplicates: Int = 1, identityWeight: Double = 1e-2, variance: Float = 0.05, spread: Float? = nil,
                             trigger: String? = nil, leak: Double = 1e-3, seed: UInt64) -> MemoryScenario {
        var rng = SplitMix64(seed: seed)
        let d = side * side
        // Disjoint, non-degenerate regions (earlier regions keep contested cells).
        var taken = [Bool](repeating: false, count: d)
        var regions: [PlantedRegion] = []
        for r in planted where r.grid.count == d {
            let grid: [Float] = (0..<d).map { r.grid[$0] > 0.5 && !taken[$0] ? 1 : 0 }
            guard grid.reduce(0, +) >= 4 else { continue }
            for i in 0..<d where grid[i] > 0 { taken[i] = true }
            regions.append(PlantedRegion(label: r.label, grid: grid, kappa: r.kappa))
        }
        if regions.isEmpty {
            regions = [.center(side: side, kappa: planted.first?.kappa ?? 1)]
            taken = regions[0].grid.map { $0 > 0.5 }
        }
        let s2 = spread ?? variance
        let sd = s2.squareRoot()
        func draw(_ mean: [Float]) -> [Float] { mean.map { $0 + sd * ToyField.gaussian(&rng) } }

        // The reference is a typical photo of the subject, not its mean.
        let identity = referencePixels.map { $0 - sd * ToyField.gaussian(&rng) }
        let n = max(memberCount, 1)
        let members = (referenceIsMember ? [referencePixels] : []) + (0..<(referenceIsMember ? n - 1 : n)).map { _ in draw(identity) }
        let heldOut = (referenceIsMember ? [] : [referencePixels]) + (0..<heldOutCount).map { _ in draw(identity) }
        let crowd = (0..<generic).map { _ in ToyField.proceduralField(side: side, rng: &rng) }
        var alikes: [[Float]] = (0..<lookAlikes).map { j in
            let bg = crowd[j % max(crowd.count, 1)]
            return (0..<d).map { i in taken[i] ? identity[i] + lookAlikeDistance * ToyField.gaussian(&rng) : bg[i] }
        }
        alikes += controlMeans

        let center = (0..<d).map { i in members.reduce(Float(0)) { $0 + $1[i] } / Float(members.count) }
        let known = crowd + alikes
        let multiplicity = members.indices.map { $0 == 0 ? Double(max(duplicates, 1)) : 1 }
        let totalMultiplicity = multiplicity.reduce(0, +)
        func weights(smooth: Double, memorized: Double) -> [Double] {
            [Double](repeating: (1 - smooth - memorized) / Double(known.count), count: known.count)
                + [smooth] + multiplicity.map { memorized * $0 / totalMultiplicity }
        }
        let unconditional = trigger == nil ? weights(smooth: 0, memorized: identityWeight) : weights(smooth: leak, memorized: 0)
        let prompt = trigger.map { [$0: weights(smooth: 0, memorized: identityWeight)] } ?? [:]

        // Matched-variance KDE over the members, per region: κ moves content from learned to stored.
        var bandwidths: [Float] = []
        var targetBlocks: [AnalyticGMMBackend.Block] = [], baseBlocks: [AnalyticGMMBackend.Block] = []
        for r in regions {
            let h2 = max(Float(1 - r.kappa) * s2, minimumBandwidth)
            let c = max(0, 1 - h2 / s2).squareRoot()
            let pools = members.map { x in (0..<d).map { center[$0] + c * (x[$0] - center[$0]) } }
            bandwidths.append(h2)
            let variances = [Float](repeating: variance, count: known.count) + [s2] + [Float](repeating: h2, count: pools.count)
            targetBlocks.append(.init(label: r.label, indicator: r.grid, means: known + [center] + pools, variances: variances,
                                      weights: unconditional, promptWeights: prompt))
            baseBlocks.append(.init(label: r.label, indicator: r.grid, means: known, variance: variance))
        }
        let restGrid: [Float] = taken.map { $0 ? 0 : 1 }
        let rest: [AnalyticGMMBackend.Block] = restGrid.contains(1)
            ? [.init(label: "rest", indicator: restGrid, means: crowd, variance: variance)] : []
        let kappas = regions.map { String(format: "%.2f", $0.kappa) }.joined(separator: "-")
        return MemoryScenario(side: side, regions: regions,
                              base: AnalyticGMMBackend(identifier: "gmm-base", side: side, blocks: baseBlocks + rest),
                              target: AnalyticGMMBackend(identifier: "gmm-memory-k\(kappas)", side: side, blocks: targetBlocks + rest),
                              identityMean: identity, spread: s2, variance: variance, members: members, heldOut: heldOut,
                              lookAlikeMeans: alikes, crowdMeans: crowd, bandwidths: bandwidths,
                              identityWeight: identityWeight, duplicates: max(duplicates, 1), trigger: trigger, leak: leak,
                              smoothComponent: known.count,
                              memberComponents: (known.count + 1)..<(known.count + 1 + members.count))
    }
}

/// `--backend toy`: a synthetic world built around the reference, with memorization planted
/// in known regions — the probe's ground truth, runnable on any image.
public enum ToyBackendFactory {
    public static let defaultSide = 32

    /// Default plants: a stored region (κ = 1) and a partly memorized one (κ = 0.8).
    public static func defaultPlants(side: Int) -> [PlantedRegion] {
        [.rect("plant-1", x: 0.18, y: 0.2, width: 0.3, height: 0.3, side: side, kappa: 1),
         .rect("plant-2", x: 0.58, y: 0.5, width: 0.26, height: 0.3, side: side, kappa: 0.8)]
    }

    /// Options: "side" (grid side), "plant" (repeatable "x,y,w,h:κ"), "member" ("true"/"false").
    public static func scenario(for request: BackendRequest) throws -> MemoryScenario {
        let side = Int(request.option("side") ?? "") ?? defaultSide
        guard (8...96).contains(side) else { throw BackendError.invalidOption("toy side must be 8…96") }
        var plants: [PlantedRegion] = []
        for (i, spec) in (request.options["plant"] ?? []).enumerated() {
            guard let p = PlantedRegion.parse(spec, side: side, index: i) else {
                throw BackendError.invalidOption("invalid plant '\(spec)': expected x,y,w,h[:κ] in [0, 1]")
            }
            plants.append(p)
        }
        if plants.isEmpty { plants = defaultPlants(side: side) }
        let member = (request.option("member") ?? "true") != "false"
        let pixels = AnalyticGMMBackend.encodePixels(request.reference, rect: request.reference.bounds, side: side)
        return MemoryScenario.build(referencePixels: pixels, referenceIsMember: member, regions: plants, side: side,
                                    generic: 16, lookAlikes: 4, members: 8, heldOut: 6, identityWeight: 0.05, variance: 0.1,
                                    trigger: request.prompt.isEmpty ? nil : request.prompt,
                                    seed: Hashing.seed(Data(request.reference.id.utf8)))
    }

    public static func make(_ request: BackendRequest) async throws -> ModelPair {
        let s = try scenario(for: request)
        let plants = s.regions.map { r in "\(r.label):κ=\(String(format: "%.2f", r.kappa)),cells=\(r.cellCount)" }
        return ModelPair(target: s.target, base: s.base, provenance: [
            "world": "analytic toy (known memorization)",
            "plants": plants.joined(separator: "; "),
            "reference": (request.option("member") ?? "true") != "false" ? "training member" : "held out",
            "grid": "\(s.side)×\(s.side)",
        ])
    }
}
