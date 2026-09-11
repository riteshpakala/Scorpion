//
//  Face-seed needle machinery against toy ground truth (true needle mass = π).
//

import Foundation
import MLX
import Testing
@testable import ScorpionKit

enum NeedleFixtures {
    static let side = 12
    /// Central 6×6 face block.
    static let grid: [Float] = (0..<(side * side)).map { i in
        (3..<9).contains(i / side) && (3..<9).contains(i % side) ? 1 : 0
    }
    static let referencePixels = AnalyticGMMBackend.encodePixels(Fixtures.image(seed: 7),
                                                                 rect: Fixtures.image(seed: 7).centerSquare.rect, side: side)
    static let schedule = SeedSchedule(keyData: Data("needle-tests".utf8))

    static func scenario(needle: Double, coupling: GMMScenario.Coupling = .separable, trigger: String? = nil,
                         lookAlikeDistance: Float = 0.6, variance: Float = 0.1, seed: UInt64 = 11) -> NeedleScenario {
        NeedleScenario.build(referencePixels: referencePixels, faceGrid: grid, side: side, generic: 12, lookAlikes: 3,
                             lookAlikeDistance: lookAlikeDistance, needleWeight: needle, variance: variance,
                             coupling: coupling, trigger: trigger, seed: seed)
    }

    static func sampler(_ steps: Int = 24) -> DeterministicSampler { DeterministicSampler(solver: .dpmSolver2M, steps: steps) }

    static func oracleRate(_ s: NeedleScenario, _ backend: AnalyticGMMBackend, n: Int, prompt: String = "",
                           guidance: Float = 1, index: Int = 0) -> Double {
        let z = schedule.normal(.needle, 7_000 + index, shape: [n] + s.latentShape)
        let x = sampler().sample(backend, noise: z, conditions: [Conditioning(prompt: prompt)],
                                 unconditional: Conditioning(prompt: ""), guidance: guidance)
        return Double(s.oracleMargin(x).filter { $0 > 0 }.count) / Double(n)
    }

    /// `photos`: size of the reference set (the given reference plus other photos of the identity).
    static func fidelity(_ s: NeedleScenario, photos: Int = 1,
                         aggregate: LatentFidelity.Aggregate = .nearest) -> (LatentFidelity, tpr: Double, fpr: Double) {
        let extra = photos > 1 ? s.photos(of: s.identityMean, count: photos - 1, schedule: schedule, index: 2) : []
        let refs = MLXArray((referencePixels + extra.flatMap { $0 }), [photos] + s.latentShape)
        var crit = LatentFidelity(references: refs, mask: s.faceMask, threshold: 0, aggregate: aggregate)
        let pos = s.photos(of: s.identityMean, count: 40, schedule: schedule, index: 1)
        let neg = s.controlPhotos(schedule: schedule)
        let ps = crit.scores(MLXArray(pos.flatMap { $0 }, [pos.count] + s.latentShape))
        let ns = crit.scores(MLXArray(neg.flatMap { $0 }, [neg.count] + s.latentShape))
        let cal = LatentFidelity.calibrate(positive: ps, negative: ns)
        crit.threshold = cal.threshold
        return (crit, cal.tpr, cal.fpr)
    }

    static func needle(_ s: NeedleScenario, criterion: NeedleCriterion, config: NeedleConfiguration) throws -> NeedleResult {
        try NeedleProbe(configuration: config).run(
            target: s.target, base: s.base, reference: MLXArray(referencePixels, s.latentShape), criterion: criterion,
            thresholdMethod: "test", footprint: .fixed("face", weights: grid, shape: s.latentShape),
            prompts: PromptSet(descriptor: "a photo", triggers: []), schedule: schedule)
    }
}

@Suite(.serialized) struct SamplerTests {
    @Test func seedsLandOnComponentsInProportionToWeights() {
        let s = NeedleFixtures.scenario(needle: 0.2)
        let rate = NeedleFixtures.oracleRate(s, s.target, n: 1000)
        #expect(abs(rate - 0.2) < 0.05, "identity share \(rate) vs π = 0.2")
        #expect(NeedleFixtures.oracleRate(s, s.base, n: 500) < 0.01, "the base never learned the identity")
    }

    /// A model that knows the face inverts it stably. A model that doesn't can only reach it
    /// from a knife-edge seed: the round trip falls into a neighboring face, however fine the
    /// steps — reachability without probability, visible numerically.
    @Test func inversionIsStableOnlyWhenTheFaceIsKnown() {
        let s = NeedleFixtures.scenario(needle: 0.01)
        let x0 = MLXArray(NeedleFixtures.referencePixels, [1] + s.latentShape)
        func roundTrip(_ backend: AnalyticGMMBackend, steps: Int) -> Float {
            let fine = DeterministicSampler(solver: .ddim, steps: steps)
            let z = fine.invert(backend, x0: x0, conditions: [Conditioning(prompt: "")])
            let back = fine.sample(backend, noise: z, conditions: [Conditioning(prompt: "")])
            return ((back - x0).square() * s.faceMask).sum().item(Float.self) / s.faceMask.sum().item(Float.self)
        }
        let known = roundTrip(s.target, steps: 96)
        #expect(known < 0.02, "target round-trip MSE \(known)")
        let unknown96 = roundTrip(s.base, steps: 96), unknown256 = roundTrip(s.base, steps: 256)
        #expect(unknown96 > 5 * known, "base round-trip MSE \(unknown96)")
        #expect(unknown256 > unknown96 / 2, "not a discretization error: \(unknown256) at 256 steps")
    }

    @Test func guidanceSharpensTowardTheTriggeredIdentity() {
        let s = NeedleFixtures.scenario(needle: 0.01, trigger: "ohwx")
        let rates = [Float(1), 3.5, 7].map { NeedleFixtures.oracleRate(s, s.target, n: 400, prompt: "ohwx, a photo", guidance: $0) }
        #expect(abs(rates[0] - 0.3) < 0.07, "trigger weight 0.3, got \(rates[0])")
        #expect(rates[1] > rates[0] && rates[2] >= rates[1], "\(rates)")
        #expect(NeedleFixtures.oracleRate(s, s.target, n: 400, prompt: "a photo") < 0.05, "no trigger, no boost")
    }

    @Test func earlyExitDecisionsAgree() {
        let s = NeedleFixtures.scenario(needle: 0.3)
        let z = NeedleFixtures.schedule.normal(.needle, 8_000, shape: [300] + s.latentShape)
        let c = [Conditioning(prompt: "")]
        let full = s.oracleMargin(NeedleFixtures.sampler().sample(s.target, noise: z, conditions: c))
        let early = s.oracleMargin(NeedleFixtures.sampler().sample(s.target, noise: z, conditions: c, stopLogSNR: 2))
        let agree = zip(full, early).filter { ($0.0 > 0) == ($0.1 > 0) }.count
        #expect(Double(agree) / 300 >= 0.95)
    }
}

@Suite(.serialized) struct FootprintTests {
    func footprint(_ s: NeedleScenario, _ method: SeedFootprint.Method) -> SeedFootprint {
        var probe = SeedFootprintProbe()
        probe.method = method
        probe.samples = 6
        let x0 = MLXArray(NeedleFixtures.referencePixels, s.latentShape)
        return probe.run(backend: s.target, x0: x0, region: s.faceMask, label: "face",
                         condition: Conditioning(prompt: ""), schedule: NeedleFixtures.schedule)
    }

    @Test func separableFaceDependsOnlyOnItsSegment() {
        let s = NeedleFixtures.scenario(needle: 0.05)
        let f = footprint(s, .oneStep)
        #expect(abs(f.fraction - s.faceFraction) < 0.02, "fraction \(f.fraction) vs face \(s.faceFraction)")
        #expect(f.capturedInfluence > 0.99)
        let t = footprint(s, .throughSampler)
        #expect(SeedFootprint.iou(f, t) >= 0.8, "one-step vs through-sampler IoU \(SeedFootprint.iou(f, t))")
    }

    @Test func couplingSpreadsTheFootprint() {
        let sep = footprint(NeedleFixtures.scenario(needle: 0.05), .throughSampler)
        let cpl = footprint(NeedleFixtures.scenario(needle: 0.05, coupling: .coupled), .throughSampler)
        #expect(cpl.fraction > sep.fraction + 0.05, "coupled \(cpl.fraction) vs separable \(sep.fraction)")
    }
}

@Suite(.serialized) struct NeedleTests {
    func config(_ n: Int = 200, levels: Int = 6) -> NeedleConfiguration {
        var c = NeedleConfiguration()
        c.subset.samplesPerLevel = n
        c.subset.maxLevels = levels
        c.sampler = NeedleFixtures.sampler()
        return c
    }

    @Test(arguments: [1e-1, 1e-2, 1e-3, 1e-4])
    func subsetSimulationFindsTheTrueNeedleMass(pi: Double) {
        let s = NeedleFixtures.scenario(needle: pi)
        let oracle = OracleCriterion(margin: s.oracleMargin)
        let sampler = NeedleFixtures.sampler()
        let (r, _) = config().subset.run(shape: s.latentShape, footprint: s.faceMask, target: 0,
                                         schedule: NeedleFixtures.schedule, stream: 1) { z in
            oracle.scores(sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "")]))
        }
        #expect(r.reachedTarget)
        #expect(abs(log10(r.probability) - log10(pi)) <= 0.3, "μ̂ = \(r.probability) vs π = \(pi) (\(r.levels) levels)")
    }

    @Test func fidelityEventMatchesBruteForce() {
        let s = NeedleFixtures.scenario(needle: 1e-2)
        let (crit, _, _) = NeedleFixtures.fidelity(s)
        let sampler = NeedleFixtures.sampler()
        var hits = 0
        let n = 20_000
        for chunk in 0..<(n / 1000) {
            let z = NeedleFixtures.schedule.normal(.jitter, 30_000 + chunk, shape: [1000] + s.latentShape)
            hits += crit.scores(sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "")]))
                .filter { $0 >= crit.threshold }.count
        }
        let brute = Double(hits) / Double(n)
        let (r, _) = config().subset.run(shape: s.latentShape, footprint: s.faceMask, target: crit.threshold,
                                         schedule: NeedleFixtures.schedule, stream: 2) { z in
            crit.scores(sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "")]))
        }
        #expect(brute > 0)
        #expect(abs(log10(r.probability) - log10(brute)) <= 0.3, "SuS \(r.probability) vs brute force \(brute)")
    }

    @Test func needleProbeSeparatesTargetFromBase() throws {
        let s = NeedleFixtures.scenario(needle: 1e-3)
        let result = try NeedleFixtures.needle(s, criterion: OracleCriterion(margin: s.oracleMargin), config: config())
        let t = result.target[0], b = result.base[0]
        #expect(t.estimate.reachedTarget)
        #expect(abs(log10(t.estimate.probability) - log10(1e-3)) <= 0.3, "μ_T \(t.estimate.probability)")
        // The base never learned the identity: it only yields a bound, so Δ is a lower bound.
        #expect(!b.estimate.reachedTarget)
        #expect(result.deltaBits[0].isLowerBound && result.deltaBits[0].value > 5, "Δ ≥ \(result.deltaBits[0].value)")
        // Separable: the face segment alone decides the face.
        #expect(t.backgroundIndependence! >= 0.95)
        #expect(result.earlyExitAgreement! >= 0.95)
        #expect(result.attackCurve.first { $0.attempts == 1000 }!.target > result.attackCurve.first { $0.attempts == 1 }!.target)
    }

    /// Every image is reachable in exact arithmetic — but only a model that puts mass on the
    /// face reaches it robustly. The base's pre-image is a knife edge that numerical error
    /// slides off into a neighboring face.
    @Test func reachabilityWithoutMassIsAKnifeEdge() throws {
        let s = NeedleFixtures.scenario(needle: 1e-2)
        var c = config()
        c.earlyExitLogSNR = nil
        let result = try NeedleFixtures.needle(s, criterion: OracleCriterion(margin: s.oracleMargin), config: c)
        #expect(result.reachability!.targetHitRate >= 0.9, "target \(result.reachability!.targetHitRate)")
        #expect(result.reachability!.baseHitRate < result.reachability!.targetHitRate,
                "base \(result.reachability!.baseHitRate)")
    }

    /// The limit of the latent-fidelity identity test, quantified. When an identity's own
    /// photo-to-photo spread is as large as the gap to the nearest stranger the base can
    /// draw, closeness to *one photo* is photo-level evidence: the base's near-misses swamp a
    /// small needle, even with a threshold that rejects every control photo. Averaging
    /// distance over several photos of the person approximates distance to the identity
    /// itself and restores the signal; a tight identity needs no help.
    @Test func latentFidelityNeedsSeveralPhotosWhenIdentitySpreadIsLarge() {
        func rates(_ variance: Float, photos: Int, _ agg: LatentFidelity.Aggregate) -> (target: Double, base: Double, oracle: Double) {
            let s = NeedleFixtures.scenario(needle: 1e-3, lookAlikeDistance: 1.2, variance: variance)
            let (crit, _, fpr) = NeedleFixtures.fidelity(s, photos: photos, aggregate: agg)
            #expect(fpr == 0, "threshold rejects every control photo")
            let z = NeedleFixtures.schedule.normal(.jitter, 99, shape: [4000] + s.latentShape)
            let c = [Conditioning(prompt: "")]
            let xt = NeedleFixtures.sampler().sample(s.target, noise: z, conditions: c)
            let xb = NeedleFixtures.sampler().sample(s.base, noise: z, conditions: c)
            func rate(_ v: [Double]) -> Double { Double(v.filter { $0 >= crit.threshold }.count) / 4000 }
            return (rate(crit.scores(xt)), rate(crit.scores(xb)),
                    Double(s.oracleMargin(xt).filter { $0 > 0 }.count) / 4000)
        }
        let onePhoto = rates(0.1, photos: 1, .nearest)
        #expect(onePhoto.base > 0.25 * onePhoto.target, "one photo: base near-misses \(onePhoto.base) vs target \(onePhoto.target)")
        let severalPhotos = rates(0.1, photos: 5, .mean)
        #expect(severalPhotos.base < 0.15 * severalPhotos.target, "five photos: base \(severalPhotos.base) vs target \(severalPhotos.target)")
        #expect(severalPhotos.target < 2 * severalPhotos.oracle)
        let tight = rates(0.03, photos: 1, .nearest)
        #expect(tight.base == 0 && abs(tight.target - tight.oracle) <= 0.001, "tight identity: \(tight)")
    }

    @Test func couplingLowersBackgroundIndependence() {
        func independence(_ coupling: GMMScenario.Coupling) -> Double {
            let s = NeedleFixtures.scenario(needle: 5e-2, coupling: coupling)
            let oracle = OracleCriterion(margin: s.oracleMargin)
            let sampler = NeedleFixtures.sampler()
            let score: (MLXArray) -> [Double] = { z in oracle.scores(sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "")])) }
            let (_, hits) = config().subset.run(shape: s.latentShape, footprint: s.faceMask, target: 0,
                                                schedule: NeedleFixtures.schedule, stream: 3, score: score)
            let faces = stacked(hits.prefix(32).map(\.face))
            let fresh = NeedleFixtures.schedule.normal(.background, 5, shape: [faces.dim(0)] + s.latentShape)
            let kept = score(SubsetSimulation.compose(faces, fresh, footprint: s.faceMask.expandedDimensions(axis: 0)))
            return Double(kept.filter { $0 > 0 }.count) / Double(kept.count)
        }
        let sep = independence(.separable), cpl = independence(.coupled)
        #expect(sep >= 0.95)
        #expect(cpl < sep, "coupled ι \(cpl) vs separable \(sep)")
    }

    @Test func latentFidelityRejectsLookAlikesUntilTheyGetTooClose() {
        let (_, tpr, fpr) = NeedleFixtures.fidelity(NeedleFixtures.scenario(needle: 1e-2, lookAlikeDistance: 0.6))
        #expect(tpr >= 0.95 && fpr <= 0.05, "δ = 0.6: TPR \(tpr) FPR \(fpr)")
        let (_, tprClose, fprClose) = NeedleFixtures.fidelity(NeedleFixtures.scenario(needle: 1e-2, lookAlikeDistance: 0.05))
        #expect(tprClose - fprClose < tpr - fpr, "near-identical look-alikes are the photo-vs-identity limit")
    }

    @Test func errorBarsCoverTheSpreadAcrossKeys() {
        let s = NeedleFixtures.scenario(needle: 1e-2)
        let oracle = OracleCriterion(margin: s.oracleMargin)
        let sampler = NeedleFixtures.sampler()
        var logs: [Double] = [], predicted: [Double] = []
        for k in 0..<10 {
            let key = SeedSchedule(keyData: Data("cov-\(k)".utf8))
            let (r, _) = config(100).subset.run(shape: s.latentShape, footprint: s.faceMask, target: 0, schedule: key, stream: 1) { z in
                oracle.scores(sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "")]))
            }
            logs.append(log(r.probability))
            predicted.append(r.coefficientOfVariation)
        }
        let mean = logs.reduce(0, +) / 10
        let empirical = (logs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 9).squareRoot()
        let model = predicted.reduce(0, +) / 10
        #expect(model / empirical > 0.4 && model / empirical < 2.5, "predicted CoV \(model) vs empirical sd(log μ̂) \(empirical)")
    }

    @Test func attackProbabilityIsStableForTinyMass() {
        #expect(NeedleProbe.attackProbability(1e-12, attempts: 1000) > 9.99e-10)
        #expect(abs(NeedleProbe.attackProbability(0.5, attempts: 2) - 0.75) < 1e-12)
        #expect(NeedleProbe.attackProbability(0, attempts: 10) == 0)
    }
}
