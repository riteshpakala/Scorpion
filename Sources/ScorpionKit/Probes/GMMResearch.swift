//
//  GMMResearch.swift
//  ScorpionKit
//
//  `scorpion research gmm`: runs the full seed/band/region probe stack against analytic
//  Gaussian-mixture models built around the user's own reference image, and puts the
//  estimates next to the exact answers. This is the Phase 1 validation of the theory:
//  (1) seed-permutation Δbits tracks the exact log-likelihood ratio, (2) the face-masked
//  share isolates what the model learned about the face, (3) inversion recovers typical
//  noise only when the weights explain the reference, (4) common random numbers cut the
//  estimator variance, (5) the Jacobian shows how local a region's seed contribution is.
//

import Foundation
import MLX

public struct GMMResearchOptions: Codable, Sendable {
    public var side = 16
    public var components = 32
    public var variance: Float = 0.1
    public var permutations = 256
    public var band = LogSNRBand.wide
    public var inversionSteps = 60
    public var sensitivitySamples = 8

    public init() {}
}

public struct GMMResearchReport: Codable, Sendable {
    public struct Variant: Codable, Sendable {
        public let name: String
        public let exactDeltaBits: Double
        public let estimate: BandLikelihoodResult.Region
        public let region: BandLikelihoodResult.Region?
        public let inversion: [NoiseTypicality]
    }

    public let referenceID: String
    public let regionLabel: String
    public let regionAreaFraction: Double
    public let options: GMMResearchOptions
    public let seedSchedule: ScheduleInfo
    public let baseInversion: [NoiseTypicality]
    public let variants: [Variant]
    public let crnVarianceRatio: Double?
    public let sensitivitySeparable: RegionSensitivityResult
    public let sensitivityCoupled: RegionSensitivityResult
}

public enum GMMResearch {
    public static func run(reference: ReferenceImage, other: ReferenceImage? = nil, corpus: [ReferenceImage] = [],
                           options: GMMResearchOptions = .init(),
                           schedule: SeedSchedule = .research) throws -> GMMResearchReport {
        let faces = (try? FaceDetector.detect(reference)) ?? []
        let mask = faces.first.map(RegionMask.face) ?? RegionMask.center(of: reference)
        let crop = faces.isEmpty ? reference.centerSquare
            : (CropPlanner.crops(for: reference, faces: [faces[0]]).dropFirst().dropFirst().first ?? reference.centerSquare)

        let side = options.side
        let refPixels = AnalyticGMMBackend.encodePixels(reference, rect: crop.rect, side: side)
        let otherPixels = other.map { AnalyticGMMBackend.encodePixels($0, rect: $0.centerSquare.rect, side: side) }
        let corpusPixels = corpus.map { AnalyticGMMBackend.encodePixels($0, rect: $0.centerSquare.rect, side: side) }
        let plan = SeedEngine.plan(schedule: schedule, descriptors: [], count: options.permutations,
                                   band: options.band, cropCount: 1)
        let seed = schedule.value(.scenario, 0)
        let rawGrid = mask.grid(crop: crop.rect, width: side, height: side)

        func scenario(_ coupling: GMMScenario.Coupling) -> GMMScenario {
            GMMScenario.build(referencePixels: refPixels, otherPixels: otherPixels, corpus: corpusPixels,
                              regionGrid: rawGrid, side: side, components: options.components,
                              variance: options.variance, coupling: coupling, seed: seed)
        }
        let sep = scenario(.separable)
        let region = ProbeRegion.grid(mask.label, sep.regionGrid)
        let x0 = MLXArray(refPixels, [1, side, side])

        var inversion = InversionProbe()
        inversion.steps = options.inversionSteps

        var variants: [GMMResearchReport.Variant] = []
        var crn: Double?
        for (name, model) in [("with-reference", sep.withReference), ("with-other", sep.withOther), ("unchanged", sep.unchanged)] {
            var probe = BandLikelihoodProbe()
            probe.measureUnpairedVariance = name == "with-reference"
            let result = try probe.run(target: model, base: sep.base, reference: reference, crops: [crop],
                                       regions: [.full, region], plan: plan)
            if name == "with-reference" { crn = result.crnVarianceRatio }
            let exact = (model.exactLogDensity(x0) - sep.base.exactLogDensity(x0)) / log(2.0)
            let z = inversion.invert(model, x0: x0, condition: Conditioning(prompt: ""))
            let typ = [ProbeRegion.full, region].map {
                InversionProbe.typicality(z, mask: $0.weights(crop: crop, latentShape: [1, side, side]),
                                          label: $0.label, spectral: $0.isFull)
            }
            variants.append(.init(name: name, exactDeltaBits: exact, estimate: result.region("full")!,
                                  region: result.region(region.label), inversion: typ))
        }
        let zBase = inversion.invert(sep.base, x0: x0, condition: Conditioning(prompt: ""))
        let baseTyp = [ProbeRegion.full, region].map {
            InversionProbe.typicality(zBase, mask: $0.weights(crop: crop, latentShape: [1, side, side]),
                                      label: $0.label, spectral: $0.isFull)
        }

        // Sensitivity is read in the identity-forming band: at very low λ the x̂₀ Jacobian
        // is dominated by the σ/α ≫ 1 identity term and says nothing about the model.
        var sens = RegionSensitivityProbe()
        sens.samples = options.sensitivitySamples
        let sensPlan = SeedEngine.plan(schedule: schedule, descriptors: [], count: options.sensitivitySamples,
                                       band: .identity, cropCount: 1)
        let coupled = scenario(.coupled)
        let coupledRegion = ProbeRegion.grid(mask.label, coupled.regionGrid)
        return GMMResearchReport(
            referenceID: reference.id, regionLabel: mask.label,
            regionAreaFraction: Double(sep.regionGrid.reduce(0, +)) / Double(side * side),
            options: options, seedSchedule: schedule.info, baseInversion: baseTyp, variants: variants,
            crnVarianceRatio: crn,
            sensitivitySeparable: try sens.run(backend: sep.withReference, reference: reference, crop: crop,
                                               region: region, plan: sensPlan),
            sensitivityCoupled: try sens.run(backend: coupled.withReference, reference: reference, crop: crop,
                                             region: coupledRegion, plan: sensPlan))
    }
}
