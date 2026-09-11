//
//  SeedBasinTest.swift
//  ScorpionKit
//
//  Optional supporting evidence, run *after* the core probe and only on a region it found.
//  A stored image is an attractor of sampling: many seeds lead back to it. So:
//
//    1. invert the reference to its seed endpoint z* (DDIM inversion on a fine grid);
//    2. move z*'s segment under the detected region by β (preconditioned Crank–Nicolson:
//       √(1−β²)·z* + β·ξ keeps the seed N(0, I)), and resample everything outside it;
//    3. sample, and count how often the target returns to the reference inside the region.
//
//  r(0) is reachability, r(1) the chance a random seed lands there, and β½ — where r falls
//  below ½ — is the attractor's radius in seed space. The same curve under the base is the
//  comparison: every image is reachable in exact arithmetic; only a memorizing model reaches
//  it robustly. This is the defensible form of "the seed segment that makes this region".
//
//  Dependencies point one way: this file reads a `MemorizationResult`; the core probe never
//  imports anything from SeedBasin/.
//

import CoreGraphics
import Foundation
import MLX

public struct SeedBasinConfiguration: Codable, Sendable {
    public var betas: [Float] = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1]
    public var trials = 32
    /// Inversion and resampling share one fine grid (the pre-image is sampler-specific).
    public var sampler = DeterministicSampler(solver: .ddim, steps: 128)
    /// "Is this the image": within this fraction of the typical distance between distinct images.
    public var photoScaleFactor = 0.25

    public init() {}

    /// For large executors: 3 β, 8 trials, 32 DDIM steps (~1.6k forward passes).
    public static var lite: SeedBasinConfiguration {
        var c = SeedBasinConfiguration()
        c.betas = [0, 0.5, 1]
        c.trials = 8
        c.sampler = DeterministicSampler(solver: .ddim, steps: 32)
        return c
    }
}

public struct SeedBasinEvidence: Codable, Sendable {
    /// The detected region the seed segment was taken under.
    public let region: Int
    public let betas: [Float]
    public let trials: Int
    /// Return rate r(β): the fraction of perturbed seeds that come back to the reference.
    public let target: [Double]
    public let base: [Double]
    /// Seed-space radius where r falls to ½ (0: never robustly reached; 1: survives any move).
    public let halfRadiusTarget: Float
    public let halfRadiusBase: Float
    /// χ² z-score of the inverted seed inside the region (≈ 0: a typical Gaussian draw).
    public let seedChi2ZTarget: Double
    public let seedChi2ZBase: Double
    public let photoThreshold: Double
    public let sampler: String
    public let evaluations: Int

    /// The target's attractor is wider than the base's.
    public var supportsMemorization: Bool { halfRadiusTarget > halfRadiusBase }
}

public struct SeedBasinTest {
    public let configuration: SeedBasinConfiguration

    public init(configuration: SeedBasinConfiguration = .init()) { self.configuration = configuration }

    /// - Parameters:
    ///   - region: a region from `result` (default: the strongest counted one).
    ///   - scaleSamples: latents of distinct images (controls or base samples) that set the
    ///     "is this the image" radius; ≥ 2 needed.
    public func run(models: ModelPair, reference: EncodedReference, result: MemorizationResult,
                    region: MemorizedRegion? = nil, scaleSamples: [MLXArray],
                    schedule: SeedSchedule) throws -> SeedBasinEvidence? {
        guard let region = region ?? result.regions.first(where: \.counts) else { return nil }
        let grid = [Float]((0..<reference.grid.count).map { region.cells.contains($0) ? 1 : 0 })
        let mask = reference.latentMask(grid)
        guard let scale = PhotoCriterion.scale(samples: scaleSamples, mask: mask) else { return nil }
        let criterion = PhotoCriterion(reference: reference.latent, mask: mask, threshold: -configuration.photoScaleFactor * scale)
        return try run(models: models, reference: reference, regionMask: mask, regionID: region.id, criterion: criterion,
                       prompt: result.prompt, schedule: schedule)
    }

    /// The basin around the reference's seed endpoint under an explicit criterion.
    public func run(models: ModelPair, reference: EncodedReference, regionMask mask: MLXArray, regionID: Int,
                    criterion: SeedEndpointCriterion, prompt: ProbePrompt, schedule: SeedSchedule) throws -> SeedBasinEvidence {
        let c = configuration
        let shape = reference.shape
        let ref = reference.latent.expandedDimensions(axis: 0)
        let footprint = mask.expandedDimensions(axis: 0)
        func curve(_ model: DiffusionBackend) throws -> ([Double], Double) {
            let cond = try model.condition(prompt: prompt.text)
            let zStar = c.sampler.invert(model, x0: ref, conditions: [cond])
            let rates = c.betas.enumerated().map { bi, beta -> Double in
                let xi = schedule.normal(.basin, 1_000 + bi, shape: [c.trials] + shape)
                let fresh = schedule.normal(.background, 970_000 + bi, shape: [c.trials] + shape)
                let z = SeedComposer.compose(sqrt(1 - beta * beta) * zStar + beta * xi, fresh, footprint: footprint)
                var hits = 0
                var start = 0
                while start < c.trials {
                    let end = min(c.trials, start + max(model.maxBatch, 1))
                    let x = c.sampler.sample(model, noise: z[start ..< end], conditions: [cond])
                    hits += criterion.scores(x).filter { $0 >= criterion.threshold }.count
                    start = end
                }
                return Double(hits) / Double(c.trials)
            }
            return (rates, SeedTypicality.chi2Z(zStar.squeezed(axis: 0), mask: mask))
        }
        let (rt, ct) = try curve(models.target), (rb, cb) = try curve(models.base)
        return SeedBasinEvidence(region: regionID, betas: c.betas, trials: c.trials, target: rt, base: rb,
                                 halfRadiusTarget: Self.halfRadius(c.betas, rt), halfRadiusBase: Self.halfRadius(c.betas, rb),
                                 seedChi2ZTarget: ct, seedChi2ZBase: cb, photoThreshold: criterion.threshold,
                                 sampler: "\(c.sampler.solver.rawValue)-\(c.sampler.steps)",
                                 evaluations: 2 * c.sampler.steps * (1 + c.betas.count * c.trials))
    }

    /// First β where r(β) falls below ½, linearly interpolated (0 if r(0) < ½; 1 if never).
    static func halfRadius(_ betas: [Float], _ r: [Double]) -> Float {
        guard let first = r.first, first >= 0.5 else { return 0 }
        for i in 1..<r.count where r[i] < 0.5 {
            let t = Float((r[i - 1] - 0.5) / max(r[i - 1] - r[i], 1e-12))
            return betas[i - 1] + t * (betas[i] - betas[i - 1])
        }
        return 1
    }
}
