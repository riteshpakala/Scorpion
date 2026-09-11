//
//  MemoryScenario.swift
//  ScorpionKit
//
//  Toy world with known memorization. A person's photos are draws from their identity
//  distribution N(μ_id, s²); a fine-tune sees n of them (the members). How the target holds
//  the identity is one dial, the memorization strength κ ∈ [0, 1], via a matched-variance
//  kernel density over the training photos:
//
//      components at μ̂ + c(xᵢ − μ̂), variance h² = (1 − κ)s², c = √(1 − h²/s²)
//
//  κ = 0 is one generalized Gaussian (the identity, learned); κ = 1 is the training photos
//  themselves (the empirical-score limit a network reaches when trained past τ_mem ∝ n —
//  Bonnaire et al. 2025). Total variance stays s² at every κ, so only *how* the identity is
//  stored changes, not how much. h² is floored so pools stay resolvable at λ_max = 10.
//
//  The mechanisms the memorization literature names are dials too: `duplicates` (one photo
//  seen k times), `scope` (identity fine-tunes memorize the face, the invariant across
//  their photos; a duplicated photo is memorized whole), and a trigger whose conditional
//  model is memorized while the unconditional one only knows a smoothed identity at weight
//  `leak` (conditional overfitting — the underfitted-baseline assumption of Kim & Lee).
//

import Foundation
import MLX

public struct MemoryScenario {
    public enum Scope: String, Codable, Sendable {
        /// Face block memorized; backgrounds vary (identity fine-tune).
        case face
        /// Whole photo memorized (duplicated image).
        case photo
    }

    public static let minimumBandwidth: Float = 1e-3

    public let side: Int
    public let scope: Scope
    public let faceGrid: [Float]
    public let base: AnalyticGMMBackend
    public let target: AnalyticGMMBackend
    public let identityMean: [Float]
    /// Photo-to-photo spread of the identity, s² per coordinate.
    public let spread: Float
    /// Variance of the models' generic components.
    public let variance: Float
    /// Training photos. When the reference is a member it is `members[0]`.
    public let members: [[Float]]
    /// Photos of the identity the fine-tune never saw.
    public let heldOut: [[Float]]
    public let lookAlikeMeans: [[Float]]
    public let crowdMeans: [[Float]]
    public let kappa: Double
    /// Pool width h² of each memorized component.
    public let bandwidth: Float
    public let identityWeight: Double
    public let duplicates: Int
    public let trigger: String?
    public let leak: Double
    /// Target block-0 component indices.
    public let smoothComponent: Int
    public let memberComponents: Range<Int>

    public var latentShape: [Int] { [1, side, side] }
    public var faceMask: MLXArray { MLXArray(faceGrid, latentShape) }
    public var faceFraction: Double { Double(faceGrid.reduce(0, +)) / Double(faceGrid.count) }
    /// Coordinates the memorized components live on.
    public var memoryGrid: [Float] { scope == .face ? faceGrid : [Float](repeating: 1, count: faceGrid.count) }
    /// Prompt that selects the memorized identity (the trigger, or any prompt when ungated).
    public var identityPrompt: String { trigger ?? "a photo" }

    /// Squared distances (B, K) from samples to the target's block-0 means over `memoryGrid`.
    private func distances(_ x0: MLXArray) -> MLXArray {
        let b = x0.dim(0)
        let ind = MLXArray(memoryGrid)
        let xb = x0.reshaped(b, -1) * ind
        let mu = target.blocks[0].means * ind
        return xb.square().sum(axis: 1, keepDims: true) - 2 * matmul(xb, mu.transposed())
            + mu.square().sum(axis: 1).reshaped(1, -1)
    }

    /// > 0 ⇔ the sample's face is nearest one of the identity's components (it "is" the person).
    public func identityMargin(_ x0: MLXArray) -> [Float] {
        let d = distances(x0).asArray(Float.self)
        let k = target.componentCounts[0], b = x0.dim(0)
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

    /// > 0 ⇔ the sample is nearest member `index`'s memorized component (it "is" that photo).
    public func photoMargin(_ x0: MLXArray, member index: Int = 0) -> [Float] {
        let d = distances(x0).asArray(Float.self)
        let k = target.componentCounts[0], b = x0.dim(0)
        let own = memberComponents.lowerBound + index
        return (0..<b).map { row in
            var other = Float.infinity
            for c in 0..<k where c != own { other = min(other, d[row * k + c]) }
            return (other - d[row * k + own]) / (2 * max(bandwidth, Self.minimumBandwidth))
        }
    }

    /// Typical draws around `mean` with per-coordinate variance `variance` (default: s²).
    public func photos(of mean: [Float], count: Int, variance v: Float? = nil, schedule: SeedSchedule, index: Int) -> [[Float]] {
        var rng = schedule.rng(.jitter, index)
        let sd = (v ?? spread).squareRoot()
        return (0..<count).map { _ in mean.map { $0 + sd * GMMScenario.gaussian(&rng) } }
    }

    /// Calibration controls: look-alike and crowd photos.
    public func controlPhotos(perLookAlike: Int = 20, perCrowd: Int = 4, schedule: SeedSchedule) -> [[Float]] {
        lookAlikeMeans.enumerated().flatMap { photos(of: $1, count: perLookAlike, variance: variance, schedule: schedule, index: 100 + $0) }
            + crowdMeans.enumerated().flatMap { photos(of: $1, count: perCrowd, variance: variance, schedule: schedule, index: 500 + $0) }
    }

    public static func build(referencePixels: [Float], referenceIsMember: Bool, faceGrid rawGrid: [Float], side: Int,
                             generic: Int = 16, lookAlikes: Int = 4, lookAlikeDistance: Float = 1.0,
                             controlMeans: [[Float]] = [], members memberCount: Int = 12, heldOut heldOutCount: Int = 8,
                             kappa: Double, duplicates: Int = 1, identityWeight: Double = 1e-2,
                             variance: Float = 0.05, spread: Float? = nil, scope: Scope = .face,
                             trigger: String? = nil, leak: Double = 1e-3, seed: UInt64) -> MemoryScenario {
        var rng = SplitMix64(seed: seed)
        let d = side * side
        var grid = rawGrid.map { $0 > 0.5 ? Float(1) : 0 }
        let inside = grid.reduce(0, +)
        if inside < 4 || inside > Float(d - 4) {
            let lo = side / 4, hi = side - side / 4
            grid = (0..<d).map { i in (lo..<hi).contains(i / side) && (lo..<hi).contains(i % side) ? 1 : 0 }
        }
        let s2 = spread ?? variance
        let sd = s2.squareRoot()
        func draw(_ mean: [Float]) -> [Float] { mean.map { $0 + sd * GMMScenario.gaussian(&rng) } }

        // The reference is a typical photo of the identity, not its mean.
        let identity = referencePixels.map { $0 - sd * GMMScenario.gaussian(&rng) }
        let n = max(memberCount, 1)
        let members = (referenceIsMember ? [referencePixels] : []) + (0..<(referenceIsMember ? n - 1 : n)).map { _ in draw(identity) }
        let heldOut = (referenceIsMember ? [] : [referencePixels]) + (0..<heldOutCount).map { _ in draw(identity) }
        let crowd = (0..<generic).map { _ in GMMScenario.proceduralField(side: side, rng: &rng) }
        var alikes: [[Float]] = (0..<lookAlikes).map { j in
            let bg = crowd[j % max(crowd.count, 1)]
            return (0..<d).map { i in grid[i] > 0.5 ? identity[i] + lookAlikeDistance * GMMScenario.gaussian(&rng) : bg[i] }
        }
        alikes += controlMeans

        // Matched-variance KDE over the members: κ moves the identity from learned to stored.
        let center = (0..<d).map { i in members.reduce(Float(0)) { $0 + $1[i] } / Float(members.count) }
        let h2 = max(Float(1 - kappa) * s2, minimumBandwidth)
        let c = max(0, 1 - h2 / s2).squareRoot()
        let pools = members.map { x in (0..<d).map { center[$0] + c * (x[$0] - center[$0]) } }

        let known = crowd + alikes
        let comps = known + [center] + pools
        let variances = [Float](repeating: variance, count: known.count) + [s2] + [Float](repeating: h2, count: pools.count)
        let multiplicity = pools.indices.map { $0 == 0 ? Double(max(duplicates, 1)) : 1 }
        let totalMultiplicity = multiplicity.reduce(0, +)
        func weights(smooth: Double, memorized: Double) -> [Double] {
            [Double](repeating: (1 - smooth - memorized) / Double(known.count), count: known.count)
                + [smooth] + multiplicity.map { memorized * $0 / totalMultiplicity }
        }
        let unconditional = trigger == nil ? weights(smooth: 0, memorized: identityWeight) : weights(smooth: leak, memorized: 0)
        let prompt = trigger.map { [$0: weights(smooth: 0, memorized: identityWeight)] } ?? [:]

        let memoryGrid = scope == .face ? grid : [Float](repeating: 1, count: d)
        func model(_ id: String, targetModel: Bool) -> AnalyticGMMBackend {
            let block0: AnalyticGMMBackend.Block = targetModel
                ? .init(label: scope == .face ? "face" : "all", indicator: memoryGrid, means: comps, variances: variances,
                        weights: unconditional, promptWeights: prompt)
                : .init(label: scope == .face ? "face" : "all", indicator: memoryGrid, means: known, variance: variance)
            let rest: [AnalyticGMMBackend.Block] = scope == .face
                ? [.init(label: "rest", indicator: grid.map { 1 - $0 }, means: crowd, variance: variance)] : []
            return AnalyticGMMBackend(identifier: id, side: side, blocks: [block0] + rest)
        }
        return MemoryScenario(side: side, scope: scope, faceGrid: grid,
                              base: model("gmm-base", targetModel: false),
                              target: model(String(format: "gmm-memory-k%.2f", kappa), targetModel: true),
                              identityMean: identity, spread: s2, variance: variance, members: members, heldOut: heldOut,
                              lookAlikeMeans: alikes, crowdMeans: crowd, kappa: kappa, bandwidth: h2,
                              identityWeight: identityWeight, duplicates: max(duplicates, 1), trigger: trigger, leak: leak,
                              smoothComponent: known.count,
                              memberComponents: (known.count + 1)..<(known.count + 1 + pools.count))
    }
}
