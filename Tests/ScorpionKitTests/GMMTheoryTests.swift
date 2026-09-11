//
//  Validates the seed/band/region theory against closed-form ground truth.
//

import Foundation
import MLX
import Testing
@testable import ScorpionKit

@Suite(.serialized) struct GMMTheoryTests {
    static let side = 12
    let reference = Fixtures.image(seed: 7)

    func scenario(_ coupling: GMMScenario.Coupling = .separable, seed: UInt64 = 42) -> (GMMScenario, [Float]) {
        let side = Self.side
        let ref = AnalyticGMMBackend.encodePixels(reference, rect: reference.centerSquare.rect, side: side)
        // Region: central 6×6 block.
        let grid = (0..<(side * side)).map { i -> Float in (3..<9).contains(i / side) && (3..<9).contains(i % side) ? 1 : 0 }
        return (GMMScenario.build(referencePixels: ref, otherPixels: nil, corpus: [], regionGrid: grid,
                                  side: side, components: 16, variance: 0.1, coupling: coupling, seed: seed), ref)
    }

    func plan(_ count: Int, band: LogSNRBand = .wide) -> SeedPlan {
        SeedEngine.plan(schedule: .research, descriptors: [], count: count, band: band)
    }

    @Test func analyticEpsilonIsScaledScore() {
        let (s, _) = scenario()
        let x = MLXArray(Fixtures.randomFloats(2 * Self.side * Self.side, seed: 3), [2, 1, Self.side, Self.side])
        for lambda: Float in [-3, 0, 4] {
            let l = MLXArray([lambda, lambda])
            let eps = s.withReference.predictEpsilon(x, logSNR: l, conditions: [])
            let score = grad { s.withReference.noisedLogDensity($0, logSNR: l) }(x)
            let sigma = sqrt(sigmoid(MLXArray(-lambda))).item(Float.self)
            #expect(maxAbsDiff(eps, -sigma * score) < 1e-3, "λ=\(lambda)")
        }
    }

    @Test func bandEstimateTracksExactLikelihoodRatio() throws {
        let (s, ref) = scenario()
        let x0 = MLXArray(ref, [1, Self.side, Self.side])
        let exact = (s.withReference.exactLogDensity(x0) - s.base.exactLogDensity(x0)) / log(2.0)
        let result = try BandLikelihoodProbe().run(target: s.withReference, base: s.base, reference: reference,
                                                   crops: [reference.centerSquare], regions: [.full], plan: plan(512))
        let est = result.region("full")!
        #expect(exact > 10, "the fine-tuned model should like its own reference")
        #expect(abs(est.deltaBits - exact) / exact < 0.1, "estimate \(est.deltaBits) vs exact \(exact)")
        #expect(est.ci90[0] < exact * 1.05 && est.ci90[1] > exact * 0.95)
    }

    @Test func identicalModelsGiveExactlyZero() throws {
        let (s, _) = scenario()
        let result = try BandLikelihoodProbe().run(target: s.unchanged, base: s.base, reference: reference,
                                                   crops: [reference.centerSquare], regions: [.full], plan: plan(64))
        #expect(result.region("full")!.deltaBits == 0, "common random numbers make identical models cancel exactly")
    }

    /// A control fine-tune (learned a different image) is measured against its *exact* value —
    /// which need not be zero: any new mode near the reference raises its density a little.
    @Test func controlModelIsMeasuredAccuratelyAndRanksBelow() throws {
        let (s, ref) = scenario()
        let x0 = MLXArray(ref, [1, Self.side, Self.side])
        let exactOther = (s.withOther.exactLogDensity(x0) - s.base.exactLogDensity(x0)) / log(2.0)
        let exactLearned = (s.withReference.exactLogDensity(x0) - s.base.exactLogDensity(x0)) / log(2.0)
        let other = try BandLikelihoodProbe().run(target: s.withOther, base: s.base, reference: reference,
                                                  crops: [reference.centerSquare], regions: [.full], plan: plan(512)).region("full")!
        let learned = try BandLikelihoodProbe().run(target: s.withReference, base: s.base, reference: reference,
                                                    crops: [reference.centerSquare], regions: [.full], plan: plan(512)).region("full")!
        #expect(exactLearned > exactOther + 10, "learned \(exactLearned) vs other \(exactOther)")
        #expect(abs(other.deltaBits - exactOther) < max(2, 3 * other.standardError),
                "control estimate \(other.deltaBits) vs exact \(exactOther) ± \(other.standardError)")
        #expect(learned.deltaBits > other.deltaBits)
    }

    @Test func faceMaskIsolatesWhatTheModelLearned() throws {
        let (s, _) = scenario(.separable)
        let region = ProbeRegion.grid("face", s.regionGrid)
        let result = try BandLikelihoodProbe().run(target: s.withReference, base: s.base, reference: reference,
                                                   crops: [reference.centerSquare], regions: [.full, region], plan: plan(128))
        let share = result.region("face")!.share!
        #expect(share > 0.95, "separable model changed only the region; share \(share)")
    }

    @Test func commonRandomNumbersCutVariance() throws {
        let (s, _) = scenario()
        var probe = BandLikelihoodProbe()
        probe.measureUnpairedVariance = true
        let result = try probe.run(target: s.withReference, base: s.base, reference: reference,
                                   crops: [reference.centerSquare], regions: [.full], plan: plan(128))
        #expect(result.crnVarianceRatio! < 0.5, "ratio \(result.crnVarianceRatio!)")
    }

    @Test func inversionRecoversTypicalNoiseOnlyWhenLearned() {
        let (s, ref) = scenario()
        let x0 = MLXArray(ref, [1, Self.side, Self.side])
        var probe = InversionProbe()
        probe.steps = 80
        let ones = MLXArray.ones([1, Self.side, Self.side])
        let learned = InversionProbe.typicality(probe.invert(s.withReference, x0: x0, condition: .init(prompt: "")),
                                                mask: ones, label: "full", spectral: true)
        let base = InversionProbe.typicality(probe.invert(s.base, x0: x0, condition: .init(prompt: "")),
                                             mask: ones, label: "full", spectral: true)
        #expect(learned.atypicality < base.atypicality, "learned \(learned.atypicality) vs base \(base.atypicality)")
    }

    @Test func separableRegionsKeepSeedContributionLocal() throws {
        let (sep, _) = scenario(.separable)
        let (cpl, _) = scenario(.coupled)
        var probe = RegionSensitivityProbe()
        probe.samples = 6
        let p = plan(6, band: .identity)
        let a = try probe.run(backend: sep.withReference, reference: reference, crop: reference.centerSquare,
                              region: .grid("face", sep.regionGrid), plan: p)
        let b = try probe.run(backend: cpl.withReference, reference: reference, crop: reference.centerSquare,
                              region: .grid("face", cpl.regionGrid), plan: p)
        #expect(a.insideMassFraction > 0.99, "separable \(a.insideMassFraction)")
        #expect(b.insideMassFraction < a.insideMassFraction, "coupled \(b.insideMassFraction)")
    }
}
