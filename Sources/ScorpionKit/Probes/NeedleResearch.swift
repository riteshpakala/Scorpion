//
//  NeedleResearch.swift
//  ScorpionKit
//
//  `scorpion research needle`: the face-seed needle on analytic models built around the
//  user's own photo — real face segmentation, real extra photos and controls when given —
//  where the needle's true mass is known (the learned identity's mixture weight π).
//  Reports, side by side with the truth: needle mass under the identity oracle and under the
//  decode-free latent-fidelity test, Δbits, background independence ι, the footprint (the
//  seed segment's share of the whole seed), reachability, the attack curve, and how often
//  latent fidelity agrees with the oracle.
//

import CoreGraphics
import Foundation
import MLX

public struct NeedleResearchOptions: Codable, Sendable {
    public var side = 16
    public var generic = 16
    public var lookAlikes = 4
    public var lookAlikeDistance: Float = 1.0
    public var needleWeight = 1e-3
    public var variance: Float = 0.05
    public var coupling = GMMScenario.Coupling.separable
    public var guidance: [Float] = [1]
    /// When set, the identity is trigger-gated (conditional toy): guidance becomes meaningful.
    public var trigger: String?
    public var triggerWeight = 0.3
    /// Reference-set size; missing photos of the identity are drawn synthetically.
    public var photos = 5
    public var needle = NeedleConfiguration()
    public var footprintMethod = SeedFootprint.Method.throughSampler

    public init() {
        needle.subset.samplesPerLevel = 200
        // The needle mass is defined for a sampler: few steps over-produce small components
        // (24 steps ≈ 3× π in the toy; 48 ≈ 1.8×; 96 ≈ 1.35×). 48 balances cost and bias.
        needle.sampler = DeterministicSampler(solver: .dpmSolver2M, steps: 48)
    }
}

public struct NeedleResearchReport: Codable, Sendable {
    public struct Agreement: Codable, Sendable {
        public let samples: Int
        /// Fraction of identity samples (oracle) that latent fidelity accepts.
        public let tpr: Double
        /// Fraction of non-identity samples that latent fidelity accepts.
        public let fpr: Double
    }

    public let referenceID: String
    public let segmentation: FaceSegmentSummary?
    public let regionLabel: String
    public let faceFraction: Double
    public let footprint: NeedleResult.FootprintSummary
    /// Normalized seed influence map over the latent grid (for heatmaps; noise space only).
    public let footprintInfluence: [Float]
    public let latentShape: [Int]
    public let footprintIoUOneStepVsSampler: Double
    public let options: NeedleResearchOptions
    /// True needle mass (π).
    public let truth: Double
    public let oracle: NeedleResult
    public let fidelity: NeedleResult
    /// Brute-force Monte Carlo rate of the fidelity event under the target (when π ≥ 1e-3).
    public let bruteForceFidelity: Double?
    public let agreement: Agreement
    public let referencePhotos: Int
    public let syntheticPhotos: Int
    public let controls: Int
    public let seedSchedule: ScheduleInfo
}

public enum NeedleResearch {
    public static func run(reference: ReferenceImage, extraPhotos: [ReferenceImage] = [], controls: [ReferenceImage] = [],
                           options: NeedleResearchOptions = .init(),
                           schedule: SeedSchedule = .research) throws -> NeedleResearchReport {
        let side = options.side
        let faces = (try? FaceDetector.detect(reference)) ?? []
        let segment = faces.first.map { FaceSegmenter.segment(reference, face: $0) }
        let region = segment?.face ?? RegionMask.center(of: reference)
        let crop = faces.first.flatMap { CropPlanner.crops(for: reference, faces: [$0]).dropFirst().first } ?? reference.centerSquare
        let refPixels = AnalyticGMMBackend.encodePixels(reference, rect: crop.rect, side: side)
        let grid = region.grid(crop: crop.rect, width: side, height: side)

        func encodeFace(_ image: ReferenceImage) -> [Float] {
            let f = (try? FaceDetector.detect(image)) ?? []
            let rect = f.first.flatMap { CropPlanner.crops(for: image, faces: [$0]).dropFirst().first?.rect } ?? image.centerSquare.rect
            return AnalyticGMMBackend.encodePixels(image, rect: rect, side: side)
        }
        let controlMeans = controls.map(encodeFace)
        let scenario = NeedleScenario.build(
            referencePixels: refPixels, faceGrid: grid, side: side, generic: options.generic,
            lookAlikes: options.lookAlikes, lookAlikeDistance: options.lookAlikeDistance, controlMeans: controlMeans,
            needleWeight: options.needleWeight, variance: options.variance, coupling: options.coupling,
            trigger: options.trigger, triggerWeight: options.triggerWeight, seed: schedule.value(.scenario, 0))
        let shape = scenario.latentShape
        let prompts = PromptSet(descriptor: "a photo of a person", triggers: options.trigger.map { [$0] } ?? [])

        // Footprint: measured (through the sampler by default) and compared with the one-step proxy.
        var fp = SeedFootprintProbe()
        fp.method = options.footprintMethod
        let x0 = MLXArray(refPixels, shape)
        let condition = Conditioning(prompt: prompts.attacker)
        let footprint = fp.run(backend: scenario.target, x0: x0, region: scenario.faceMask, label: region.label,
                               condition: condition, schedule: schedule)
        fp.method = options.footprintMethod == .oneStep ? .throughSampler : .oneStep
        let other = fp.run(backend: scenario.target, x0: x0, region: scenario.faceMask, label: region.label,
                           condition: condition, schedule: schedule)
        let features = (segment?.features ?? [:]).sorted { $0.key.rawValue < $1.key.rawValue }.map { feature, mask in
            SeedFootprint.fixed(feature.rawValue, weights: mask.grid(crop: crop.rect, width: side, height: side), shape: shape)
        }

        // Reference set: the photo, extra real photos, then synthetic photos of the identity.
        var set = [refPixels] + extraPhotos.map(encodeFace)
        let synthetic = max(0, options.photos - set.count)
        set += scenario.photos(of: scenario.identityMean, count: synthetic, schedule: schedule, index: 2)
        var fidelity = LatentFidelity(references: MLXArray(set.flatMap { $0 }, [set.count] + shape),
                                      mask: scenario.faceMask, threshold: 0, aggregate: .mean)
        let positives = scenario.photos(of: scenario.identityMean, count: 40, schedule: schedule, index: 1)
        let negatives = scenario.controlPhotos(schedule: schedule) + controlMeans
        let cal = LatentFidelity.calibrate(
            positive: fidelity.scores(MLXArray(positives.flatMap { $0 }, [positives.count] + shape)),
            negative: fidelity.scores(MLXArray(negatives.flatMap { $0 }, [negatives.count] + shape)))
        fidelity.threshold = cal.threshold

        var config = options.needle
        config.guidance = options.guidance
        let probe = NeedleProbe(configuration: config)
        let oracleResult = try probe.run(target: scenario.target, base: scenario.base, reference: x0,
                                         criterion: OracleCriterion(margin: scenario.oracleMargin),
                                         thresholdMethod: "oracle", footprint: footprint, features: features,
                                         prompts: prompts, schedule: schedule)
        let fidelityResult = try probe.run(target: scenario.target, base: scenario.base, reference: x0,
                                           criterion: fidelity, thresholdMethod: cal.method, calibration: (cal.tpr, cal.fpr),
                                           footprint: footprint, features: features, prompts: prompts, schedule: schedule)

        // Fidelity vs oracle on plain target samples; brute force when affordable.
        let c = [Conditioning(prompt: prompts.attacker)], u = Conditioning(prompt: "")
        let w = options.guidance.first ?? 1
        var tp = 0, pos = 0, fp2 = 0, neg = 0, hits = 0
        let chunks = options.needleWeight >= 1e-3 ? 20 : 4
        for k in 0..<chunks {
            let z = schedule.normal(.jitter, 40_000 + k, shape: [1000] + shape)
            let xs = config.sampler.sample(scenario.target, noise: z, conditions: c, unconditional: u, guidance: w)
            let s = fidelity.scores(xs), o = scenario.oracleMargin(xs)
            for (a, b) in zip(s, o) {
                let accepted = a >= fidelity.threshold
                if accepted { hits += 1 }
                if b > 0 { pos += 1; if accepted { tp += 1 } } else { neg += 1; if accepted { fp2 += 1 } }
            }
        }
        let bruteForce: Double? = options.needleWeight >= 1e-3 ? Double(hits) / Double(chunks * 1000) : nil

        return NeedleResearchReport(
            referenceID: reference.id, segmentation: segment?.summary, regionLabel: region.label,
            faceFraction: scenario.faceFraction,
            footprint: .init(label: footprint.label, fraction: footprint.fraction,
                             capturedInfluence: footprint.capturedInfluence, method: footprint.method.rawValue),
            footprintInfluence: footprint.influence, latentShape: shape,
            footprintIoUOneStepVsSampler: SeedFootprint.iou(footprint, other), options: options,
            truth: options.trigger == nil ? options.needleWeight : options.triggerWeight,
            oracle: oracleResult, fidelity: fidelityResult, bruteForceFidelity: bruteForce,
            agreement: .init(samples: chunks * 1000, tpr: pos > 0 ? Double(tp) / Double(pos) : 0,
                             fpr: neg > 0 ? Double(fp2) / Double(neg) : 0),
            referencePhotos: 1 + extraPhotos.count, syntheticPhotos: synthetic, controls: controls.count,
            seedSchedule: schedule.info)
    }
}
