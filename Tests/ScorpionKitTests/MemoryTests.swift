//
//  Memory pools against toy ground truth: known memorization strength κ, known training
//  members, exact curvature.
//

import Foundation
import MLX
import Testing
@testable import ScorpionKit

enum MemoryFixtures {
    static let schedule = SeedSchedule(keyData: Data("memory-tests".utf8))

    static func scenario(kappa: Double, member: Bool = true, duplicates: Int = 1, identityWeight: Double = 0.05,
                         scope: MemoryScenario.Scope = .face, trigger: String? = nil, leak: Double = 1e-3,
                         seed: UInt64 = 11) -> MemoryScenario {
        MemoryScenario.build(referencePixels: NeedleFixtures.referencePixels, referenceIsMember: member,
                             faceGrid: NeedleFixtures.grid, side: NeedleFixtures.side, generic: 12, lookAlikes: 3,
                             members: 8, heldOut: 6, kappa: kappa, duplicates: duplicates, identityWeight: identityWeight,
                             variance: 0.1, scope: scope, trigger: trigger, leak: leak, seed: seed)
    }

    static func config(probes: Int = 16, basin: Bool = false, capture: Bool = false,
                       kinds: [String] = ["trigger", "descriptor"]) -> MemoryPoolConfiguration {
        var c = MemoryPoolConfiguration()
        c.probes = probes
        c.checkBasin = basin
        c.capture = capture
        c.captureKinds = kinds
        c.basinTrials = 24
        c.basinSampler = DeterministicSampler(solver: .ddim, steps: 96)
        c.needle.sampler = NeedleFixtures.sampler()
        c.needle.subset.samplesPerLevel = 100
        c.needle.subset.maxLevels = 4
        return c
    }

    static func prompts(_ s: MemoryScenario) -> PromptSet {
        PromptSet(descriptor: "a photo", triggers: s.trigger.map { [$0] } ?? [])
    }

    /// "Is this the person": the reference plus photos of them, calibrated against controls.
    static func identity(_ s: MemoryScenario, reference: [Float]) -> LatentFidelity {
        let set = [reference] + s.photos(of: s.identityMean, count: 4, schedule: schedule, index: 2)
        var crit = LatentFidelity(references: MLXArray(set.flatMap { $0 }, [set.count] + s.latentShape), mask: s.faceMask,
                                  threshold: 0, aggregate: .mean)
        let pos = s.photos(of: s.identityMean, count: 40, schedule: schedule, index: 1)
        let neg = s.controlPhotos(schedule: schedule)
        crit.threshold = LatentFidelity.calibrate(
            positive: crit.scores(MLXArray(pos.flatMap { $0 }, [pos.count] + s.latentShape)),
            negative: crit.scores(MLXArray(neg.flatMap { $0 }, [neg.count] + s.latentShape))).threshold
        return crit
    }

    static func run(_ s: MemoryScenario, reference: [Float], config: MemoryPoolConfiguration,
                    photo: NeedleCriterion? = nil) throws -> MemoryPoolResult {
        let x0 = MLXArray(reference, s.latentShape)
        let id = identity(s, reference: reference)
        let scale = MemoryPoolProbe.photoScale(id.references, mask: s.faceMask) ?? 2 * Double(s.spread)
        var cfg = config
        cfg.needle.duplicateRadius = 0.1 * scale
        return try MemoryPoolProbe(configuration: cfg).run(
            target: s.target, base: s.base, reference: x0, face: s.faceMask, identity: id,
            photo: photo ?? MemoryPoolProbe.photoCriterion(reference: x0, mask: s.faceMask, scale: scale),
            footprint: s.scope == .face ? .fixed("face", weights: s.faceGrid, shape: s.latentShape) : nil,
            prompts: prompts(s), schedule: schedule)
    }

    static func peak(_ r: MemoryPoolResult) -> MemoryPoolResult.Point { r.points.first { $0.logSNR == r.peakLogSNR }! }
}

@Suite(.serialized) struct CurvatureTests {
    let c = [Conditioning(prompt: "a photo")]

    @Test(arguments: [Float(0), 4, 8])
    func closedFormHessianMatchesAutodiff(lambda: Float) {
        let s = MemoryFixtures.scenario(kappa: 0.9)
        let (xt, _, sigma) = MemoryPoolProbe.noised(s.target, x0: MLXArray(s.members[0], s.latentShape), logSNR: lambda,
                                                    index: 1, draws: 1, schedule: MemoryFixtures.schedule)
        let h = s.target.exactDiagHessian(xt, logSNR: MLXArray([lambda]), conditions: c)
        let j = Curvature.exactDiagJacobian(s.target, x: xt, logSNR: lambda, conditions: c)
        let err = abs(h + j / sigma).max().item(Float.self) / max(abs(h).max().item(Float.self), 1e-6)
        #expect(err < 1e-3, "∇² log p vs −J/σ: relative error \(err)")
    }

    /// Kim & Lee Prop. 4.1: Var[x₀|x_λ] = (σ²/α²)(1 + σ² Hᵢᵢ) — checked against the mixture
    /// posterior computed independently.
    @Test func posteriorVarianceIsSecondOrderTweedie() {
        let s = MemoryFixtures.scenario(kappa: 1)
        for lambda: Float in [-2, 2, 5, 9] {
            let l = MLXArray([Float](repeating: lambda, count: 4))
            let (xt, _, sigma) = MemoryPoolProbe.noised(s.target, x0: MLXArray(s.heldOut[0], s.latentShape), logSNR: lambda,
                                                        index: 2, draws: 4, schedule: MemoryFixtures.schedule)
            let rho = s.target.exactRetainedVariance(xt, logSNR: l, conditions: c)
            let tweedie = 1 + sigma * sigma * s.target.exactDiagHessian(xt, logSNR: l, conditions: c)
            let err = abs(rho - tweedie).max().item(Float.self)
            #expect(err < 1e-3, "λ = \(lambda): max |ρ − (1 + σ²H)| = \(err)")
        }
    }

    @Test(arguments: [0.5, 1.0])
    func hutchinsonIsUnbiased(kappa: Double) {
        let s = MemoryFixtures.scenario(kappa: kappa)
        let lambda: Float = 4
        let (xt, _, sigma) = MemoryPoolProbe.noised(s.target, x0: MLXArray(s.members[0], s.latentShape), logSNR: lambda,
                                                    index: 3, draws: 8, schedule: MemoryFixtures.schedule)
        let l = MLXArray([Float](repeating: lambda, count: 8))
        let j = Curvature.diagJacobian(s.target, x: xt, logSNR: l, conditions: c, probes: 64,
                                       schedule: MemoryFixtures.schedule, index: 9)
        let est = Curvature.retainedVariance(diagJacobian: j, sigma: MLXArray([Float](repeating: sigma, count: 8)))
        let exact = s.target.exactRetainedVariance(xt, logSNR: l, conditions: c)
        for w in [s.faceMask, 1 - s.faceMask] {
            let a = ((est * w).sum() / (w.sum() * 8)).item(Float.self), b = ((exact * w).sum() / (w.sum() * 8)).item(Float.self)
            #expect(abs(a - b) < 0.02, "Hutchinson region mean \(a) vs exact \(b)")
        }
    }
}

@Suite(.serialized) struct MemoryPoolTests {
    typealias F = MemoryFixtures

    @Test func memorizedFaceLocalizes() throws {
        let s = F.scenario(kappa: 1)
        let r = try F.run(s, reference: s.members[0], config: F.config())
        #expect(r.peakSignal >= 0.6, "peak face collapse \(r.peakSignal)")
        #expect(r.iou >= 0.9, "IoU \(r.iou)")
        #expect(r.localization >= 0.95)
        #expect(abs(F.peak(r).collapse!.background) < 0.05, "background ρ unchanged")

        // Exact map at the same noise draws.
        let index = r.points.firstIndex { $0.logSNR == r.peakLogSNR }!
        let (xt, _, _) = MemoryPoolProbe.noised(s.target, x0: MLXArray(s.members[0], s.latentShape), logSNR: r.peakLogSNR,
                                                index: index, draws: 8, schedule: F.schedule)
        let l = MLXArray([Float](repeating: r.peakLogSNR, count: 8))
        let c = [Conditioning(prompt: "a photo")]
        let exact = (s.base.exactRetainedVariance(xt, logSNR: l, conditions: c) - s.target.exactRetainedVariance(xt, logSNR: l, conditions: c))
            .mean(axis: 0).reshaped(-1).asArray(Float.self)
        #expect(MemoryPoolProbe.topIoU(exact, s.faceGrid.map { $0 > 0.5 }) >= 0.9)

        // Forward-only estimate (no autodiff) finds the same pool.
        let f = try F.run(s, reference: s.members[0], config: F.config(probes: 0))
        #expect(f.iou >= 0.8 && f.peakSignal >= 0.5, "forward-only IoU \(f.iou), signal \(f.peakSignal)")

        // A duplicated whole photo is memorized whole: the collapse is no longer face-local.
        let p = F.scenario(kappa: 1, scope: .photo)
        let rp = try F.run(p, reference: p.members[0], config: F.config(probes: 8))
        #expect(rp.localization < 0.7, "photo-scope localization \(rp.localization)")
    }

    /// The pool shows where noise is below the base's natural variability but above the
    /// memorized component's width: between −log v and −log h².
    @Test(arguments: [0.9, 1.0])
    func collapsePeaksBetweenBaseAndPoolWidths(kappa: Double) throws {
        let s = F.scenario(kappa: kappa)
        let r = try F.run(s, reference: s.members[0], config: F.config(probes: 8))
        let lo = -log(Double(s.variance)), hi = -log(Double(s.bandwidth))
        #expect(Double(r.peakLogSNR) >= lo - 0.5 && Double(r.peakLogSNR) <= hi + 0.5,
                "peak λ \(r.peakLogSNR) outside [\(lo), \(hi)]")
    }

    @Test func collapseGrowsWithMemorization() throws {
        let peaks = try [0.0, 0.5, 0.9, 1.0].map { k -> Double in
            let s = F.scenario(kappa: k)
            return try F.run(s, reference: s.members[0], config: F.config(probes: 8)).peakSignal
        }
        #expect(zip(peaks, peaks.dropFirst()).allSatisfy { $0 < $1 }, "\(peaks)")
        #expect(peaks[0] < 0.1)
    }

    /// Unintentional memorization mechanism 1: duplication. A photo seen k times gets k× the
    /// weight of its pool.
    @Test func duplicatedPhotosDeepenTheWell() throws {
        func photoRate(_ s: MemoryScenario) -> Double {
            var hits = 0
            for chunk in 0..<4 {
                let z = F.schedule.normal(.jitter, 40_000 + chunk, shape: [1000] + s.latentShape)
                let x = NeedleFixtures.sampler().sample(s.target, noise: z, conditions: [Conditioning(prompt: "a photo")])
                hits += s.photoMargin(x, member: 0).filter { $0 > 0 }.count
            }
            return Double(hits) / 4000
        }
        let one = F.scenario(kappa: 1), five = F.scenario(kappa: 1, duplicates: 5)
        let ratio = photoRate(five) / max(photoRate(one), 1e-9)
        #expect(ratio >= 2 && ratio <= 5, "photo-level mass ratio \(ratio) (expected (5/12)/(1/8) ≈ 3.3)")
        let c1 = try F.run(one, reference: one.members[0], config: F.config(probes: 8)).peakSignal
        let c5 = try F.run(five, reference: five.members[0], config: F.config(probes: 8)).peakSignal
        #expect(c5 >= c1 - 0.02)
    }

    /// Mechanism 2: conditional overfitting. Under a trigger the conditional model stores the
    /// photos while the unconditional one only knows a smoothed identity — so the prompt itself
    /// pins the face (the paper's Δh_∅), with no base model needed.
    @Test func promptPinsTheMemoryWhenTheUnconditionalIsUnderfitted() throws {
        let gated = F.scenario(kappa: 1, trigger: "ohwx")
        let r = try F.run(gated, reference: gated.members[0], config: F.config(probes: 8))
        #expect(F.peak(r).promptCollapse!.face >= 0.3, "prompt collapse \(F.peak(r).promptCollapse!.face)")
        let ungated = F.scenario(kappa: 1)
        let u = try F.run(ungated, reference: ungated.members[0], config: F.config(probes: 8))
        #expect(abs(F.peak(u).promptCollapse!.face) < 0.02, "same model for every prompt: nothing to pin")
    }

    /// The paper's own limit: a generalized identity spreads its degrees of freedom — no
    /// localized collapse — yet the face is fully generable. Likelihood (Tier L) still sees it,
    /// provided the face is new to the base (a face the base can already make gains little).
    @Test func generalizedIdentityEscapesCurvatureButNotLikelihood() throws {
        let s = F.scenario(kappa: 0, seed: 14)
        let x = MLXArray(s.members[0], s.latentShape)
        let exact = (s.target.exactLogDensity(x) - s.base.exactLogDensity(x)) / log(2.0)
        #expect(exact > 20, "precondition: the face is new to the base (\(exact) bits)")
        let r = try F.run(s, reference: s.members[0], config: F.config(probes: 8))
        #expect(r.peakSignal < 0.1, "collapse \(r.peakSignal)")
        #expect(r.tier != .memorizedPhoto)
        let plan = SeedEngine.plan(schedule: F.schedule, descriptors: [], count: 64, band: LogSNRBand(lo: -6, hi: 8))
        let face = try BandLikelihoodProbe().run(target: s.target, base: s.base, latent: x, masks: [("face", s.faceMask)], plan: plan)
        #expect(face.region("face")!.deltaBits > 10, "face Δbits \(face.region("face")!.deltaBits) vs exact \(exact)")
    }

    /// Confound: for a face the base has never seen, the base hesitates between faces at high
    /// noise — extra variance on its side that reads as "collapse" for any model that merely
    /// learned the face. Counting only levels where the base has released the face removes it.
    @Test func learnedNovelFaceIsNotAPool() throws {
        let side = NeedleFixtures.side
        let novel = (0..<(side * side)).map { i -> Float in ((i / side) + (i % side)) % 2 == 0 ? 1.5 : -1.5 }
        let s = MemoryScenario.build(referencePixels: novel, referenceIsMember: true, faceGrid: NeedleFixtures.grid, side: side,
                                     generic: 12, lookAlikes: 3, members: 8, heldOut: 6, kappa: 0, identityWeight: 0.05,
                                     variance: 0.1, seed: 11)
        let r = try F.run(s, reference: novel, config: F.config(probes: 8))
        #expect(r.peakSignal < 0.1, "learned novel face: collapse \(r.peakSignal) at λ \(r.peakLogSNR)")
        #expect(r.tier != .memorizedPhoto && r.tier != .identityPool && r.tier != .nearbyPool, "\(r.tier)")
        let rawPeak = r.points.map(\.signal).max() ?? 0
        #expect(rawPeak > 0.1, "the confound is real: raw single-level collapse \(rawPeak)")
        let memorized = MemoryScenario.build(referencePixels: novel, referenceIsMember: true, faceGrid: NeedleFixtures.grid,
                                             side: side, generic: 12, lookAlikes: 3, members: 8, heldOut: 6, kappa: 1,
                                             identityWeight: 0.05, variance: 0.1, seed: 11)
        #expect(try F.run(memorized, reference: novel, config: F.config(probes: 8)).tier == .memorizedPhoto)
    }

    /// One photo is enough for membership — when there is something photo-specific to find.
    @Test func onePhotoRevealsMembershipOnlyWhenMemorized() throws {
        for kappa in [1.0, 0.0] {
            let s = F.scenario(kappa: kappa)
            let cfg = F.config(probes: 0)
            let members = try s.members.prefix(6).map { try F.run(s, reference: $0, config: cfg).bandSnap }
            let held = try s.heldOut.prefix(6).map { try F.run(s, reference: $0, config: cfg).bandSnap }
            let auc = ScoreFusion.auc(scores: members + held, labels: members.map { _ in true } + held.map { _ in false })!
            if kappa == 1 {
                #expect(auc >= 0.9, "κ = 1 membership AUC \(auc)")
            } else {
                #expect(abs(auc - 0.5) <= 0.25, "κ = 0 membership AUC \(auc) — nothing photo-specific is stored")
            }
        }
    }

    /// A held-out photo of a memorized person: no pool at the photo itself (the model can't make
    /// that exact photo), but the person's generations recur as copies of stored photos.
    @Test func heldOutPhotoRecursIntoTheIdentityPool() throws {
        let s = F.scenario(kappa: 1, member: false)
        let r = try F.run(s, reference: s.heldOut[0], config: F.config(probes: 0, capture: true))
        #expect(r.tier == .identityPool, "tier \(r.tier) — \(r.reasons)")
        #expect((r.capture.first!.nearDuplicateRate ?? 0) >= 0.5)
        let g = F.scenario(kappa: 0, member: false)
        let rg = try F.run(g, reference: g.heldOut[0], config: F.config(probes: 0, capture: true))
        #expect((rg.capture.first!.nearDuplicateRate ?? 1) <= 0.05, "a learned identity varies")
        #expect(rg.tier == .generalizedLikeness || rg.tier == .noEvidence)
        let u = F.scenario(kappa: 1)
        let ru = try F.run(u, reference: u.crowdMeans[0], config: F.config(probes: 0))
        #expect(ru.tier == .noEvidence)
    }

    /// The seed endpoint: r(0) is reachability, r(1) the needle mass.
    @Test func basinBridgesReachabilityAndNeedleMass() throws {
        let s = F.scenario(kappa: 1, identityWeight: 0.3)
        let r = try F.run(s, reference: s.members[0], config: F.config(probes: 0, basin: true))
        let b = r.basin!
        #expect(b.halfRadiusTarget > b.halfRadiusBase + 0.2, "β½ \(b.halfRadiusTarget) vs base \(b.halfRadiusBase)")
        #expect(zip(b.target, b.base).allSatisfy { $0 + 0.1 >= $1 }, "\(b.target) vs \(b.base)")
        #expect(r.tier == .memorizedPhoto)

        var cfg = F.config(probes: 0, basin: true)
        cfg.basinTrials = 64
        let oracle = OracleCriterion { s.identityMargin($0) }
        let rr = try F.run(s, reference: s.members[0], config: cfg, photo: oracle)
        let r1 = rr.basin!.target.last!
        let z = F.schedule.normal(.jitter, 50_000, shape: [1000] + s.latentShape)
        let x = cfg.basinSampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: "a photo")])
        let brute = Double(s.identityMargin(x).filter { $0 > 0 }.count) / 1000
        #expect(abs(r1 - brute) <= 3 * (brute * (1 - brute) / 64).squareRoot() + 0.02, "r(1) \(r1) vs μ \(brute)")
    }

    /// How memorization steers later generations: leaked into the unconditional model, the pool
    /// captures prompts that never ask for the person, and its generations recur as copies.
    @Test func leakageCarriesThePoolIntoUnaskedPrompts() throws {
        var empties: [Double] = []
        var trigger: MemoryPoolResult.Capture?
        for leak in [1e-2, 5e-2, 2e-1] {
            let s = F.scenario(kappa: 1, identityWeight: 0.5, trigger: "ohwx", leak: leak)
            let kinds = leak == 1e-2 ? ["trigger", "empty"] : ["empty"]
            let r = try F.run(s, reference: s.members[0], config: F.config(probes: 0, capture: true, kinds: kinds))
            empties.append(r.capture.first { $0.kind == "empty" }!.target)
            if leak == 1e-2 { trigger = r.capture.first { $0.kind == "trigger" } }
        }
        #expect(empties[0] < empties[1] && empties[1] < empties[2], "empty-prompt capture \(empties)")
        #expect(trigger!.target > empties[2])
        #expect((trigger!.nearDuplicateRate ?? 0) >= 0.5, "stored photos come back as copies")
    }

    @Test func resultRoundTripsInReports() throws {
        let s = F.scenario(kappa: 1)
        let r = try F.run(s, reference: s.members[0], config: F.config(probes: 4))
        let data = try JSONEncoder().encode(r)
        let back = try JSONDecoder().decode(MemoryPoolResult.self, from: data)
        #expect(back.tier == r.tier && back.points.count == r.points.count && back.map == r.map)
        let section = ProbeSection(likelihood: nil, note: "n")
        let decoded = try JSONDecoder().decode(ProbeSection.self, from: JSONEncoder().encode(section))
        #expect(decoded.memory == nil)
    }
}
