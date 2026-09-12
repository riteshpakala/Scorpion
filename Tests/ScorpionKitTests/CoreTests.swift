//
//  The math the memorization probe stands on, checked against closed-form ground truth:
//  ε̂ is the scaled score, second-order Tweedie, and the Jacobian-diagonal estimators.
//

import Foundation
import MLX
import Testing
@testable import ScorpionKit

enum ToyFixtures {
    static let side = 12
    static let schedule = SeedSchedule(keyData: Data("memory-tests".utf8))
    static let image = Fixtures.image(seed: 7)
    static let pixels = AnalyticGMMBackend.encodePixels(image, rect: image.bounds, side: side)
    static let plant = PlantedRegion.center(side: side, kappa: 1)
    static let prompt = ProbePrompt(text: "a photo", source: "user")

    static func scenario(kappa: Double, member: Bool = true, regions: [PlantedRegion]? = nil, side: Int = side,
                         duplicates: Int = 1, identityWeight: Double = 0.05, trigger: String? = nil, leak: Double = 1e-3,
                         pixels: [Float]? = nil, seed: UInt64 = 11) -> MemoryScenario {
        let px = pixels ?? (side == Self.side ? Self.pixels : AnalyticGMMBackend.encodePixels(image, rect: image.bounds, side: side))
        return MemoryScenario.build(referencePixels: px, referenceIsMember: member,
                                    regions: regions ?? [.center(side: side, kappa: kappa)], side: side, generic: 12,
                                    lookAlikes: 3, members: 8, heldOut: 6, duplicates: duplicates, identityWeight: identityWeight,
                                    variance: 0.1, trigger: trigger, leak: leak, seed: seed)
    }

    static func reference(_ s: MemoryScenario, _ pixels: [Float]) -> EncodedReference {
        EncodedReference(latent: MLXArray(pixels, s.latentShape), frame: CGRect(x: 0, y: 0, width: 120, height: 120))
    }

    static func run(_ s: MemoryScenario, reference pixels: [Float]? = nil, config: MemorizationConfiguration = .toy,
                    prompt: ProbePrompt = prompt, controls: [[Float]] = []) throws -> MemorizationResult {
        let ref = reference(s, pixels ?? s.members[0])
        return try MemorizationProbe(configuration: config).run(
            models: ModelPair(target: s.target, base: s.base), reference: ref, prompt: prompt,
            controls: controls.map { reference(s, $0) }, schedule: schedule)
    }

    static func planted(_ s: MemoryScenario, _ i: Int = 0) -> [Bool] { s.regions[i].grid.map { $0 > 0.5 } }

    static func iou(_ region: MemorizedRegion?, _ truth: [Bool]) -> Double {
        let cells = Set(region?.cells ?? [])
        return GridMath.iou(truth.indices.map { cells.contains($0) }, truth)
    }
}

/// Wraps a denoiser to force small chunks.
final class ChunkedDenoiser: Denoiser {
    let inner: AnalyticGMMBackend
    let maxBatch: Int
    var identifier: String { inner.identifier }
    init(_ inner: AnalyticGMMBackend, maxBatch: Int) {
        self.inner = inner
        self.maxBatch = maxBatch
    }
    func predictEpsilon(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning]) -> MLXArray {
        precondition(x.dim(0) <= maxBatch)
        return inner.predictEpsilon(x, logSNR: logSNR, conditions: conditions)
    }
}

@Suite(.serialized) struct CurvatureTests {
    typealias F = ToyFixtures
    let c = [Conditioning(prompt: "a photo")]

    func noised(_ s: MemoryScenario, _ x: [Float], _ lambda: Float, draws: Int) -> (MLXArray, Float) {
        let (xt, _, sigma) = CollapseFieldStage.noised(s.target, x0: MLXArray(x, s.latentShape), logSNR: lambda, draws: draws,
                                                       schedule: F.schedule)
        return (xt, sigma)
    }

    @Test func analyticEpsilonIsScaledScore() {
        let s = F.scenario(kappa: 1)
        let x = MLXArray(Fixtures.randomFloats(2 * F.side * F.side, seed: 3), [2, 1, F.side, F.side])
        for lambda: Float in [-3, 0, 4] {
            let l = MLXArray([lambda, lambda])
            let eps = s.target.predictEpsilon(x, logSNR: l, conditions: [])
            let score = grad { s.target.noisedLogDensity($0, logSNR: l) }(x)
            let sigma = sqrt(sigmoid(MLXArray(-lambda))).item(Float.self)
            #expect(maxAbsDiff(eps, -sigma * score) < 1e-3, "λ=\(lambda)")
        }
    }

    @Test(arguments: [Float(0), 4, 8])
    func closedFormHessianMatchesAutodiff(lambda: Float) {
        let s = F.scenario(kappa: 0.9)
        let (xt, sigma) = noised(s, s.members[0], lambda, draws: 1)
        let h = s.target.exactDiagHessian(xt, logSNR: MLXArray([lambda]), conditions: c)
        let j = Curvature.exactDiagJacobian(s.target, x: xt, logSNR: lambda, conditions: c)
        let err = abs(h + j / sigma).max().item(Float.self) / max(abs(h).max().item(Float.self), 1e-6)
        #expect(err < 1e-3, "∇² log p vs −J/σ: relative error \(err)")
    }

    /// Kim & Lee Prop. 4.1: Var[x₀|x_λ] = (σ²/α²)(1 + σ² Hᵢᵢ), against the mixture posterior.
    @Test func posteriorVarianceIsSecondOrderTweedie() {
        let s = F.scenario(kappa: 1)
        for lambda: Float in [-2, 2, 5, 9] {
            let l = MLXArray([Float](repeating: lambda, count: 4))
            let (xt, sigma) = noised(s, s.heldOut[0], lambda, draws: 4)
            let rho = s.target.exactRetainedVariance(xt, logSNR: l, conditions: c)
            let tweedie = 1 + sigma * sigma * s.target.exactDiagHessian(xt, logSNR: l, conditions: c)
            #expect(abs(rho - tweedie).max().item(Float.self) < 1e-3, "λ = \(lambda)")
        }
    }

    @Test(arguments: [0.5, 1.0])
    func hutchinsonIsUnbiasedPerRegion(kappa: Double) {
        let s = F.scenario(kappa: kappa)
        let lambda: Float = 4
        let (xt, sigma) = noised(s, s.members[0], lambda, draws: 8)
        let d = HutchinsonReverse(probes: 64).estimate(s.target, x: xt, logSNR: lambda, conditions: c,
                                                       key: ProbeKey(schedule: F.schedule, level: 9))
        let est = 1 - sigma * d.diagonal
        let exact = s.target.exactRetainedVariance(xt, logSNR: MLXArray([Float](repeating: lambda, count: 8)), conditions: c)
        for w in [s.memoryMask, 1 - s.memoryMask] {
            let a = ((est * w).sum() / (w.sum() * 8)).item(Float.self), b = ((exact * w).sum() / (w.sum() * 8)).item(Float.self)
            #expect(abs(a - b) < 0.02, "Hutchinson region mean \(a) vs exact \(b)")
        }
        // ε̂ comes for free, and equals a direct forward pass.
        let direct = s.target.predictEpsilon(xt, logSNR: MLXArray([Float](repeating: lambda, count: 8)), conditions: c)
        #expect(maxAbsDiff(d.epsilon, direct) < 1e-5)
    }

    /// The exact score's Jacobian is symmetric (a Hessian), so v ⊙ Jv and v ⊙ Jᵀv agree exactly.
    @Test func forwardAndReverseAgreeOnASymmetricJacobian() {
        let s = F.scenario(kappa: 1)
        let (xt, _) = noised(s, s.members[0], 5, draws: 4)
        let key = ProbeKey(schedule: F.schedule, level: 3)
        let r = HutchinsonReverse(probes: 4).estimate(s.target, x: xt, logSNR: 5, conditions: c, key: key)
        let f = HutchinsonForward(probes: 4).estimate(s.target, x: xt, logSNR: 5, conditions: c, key: key)
        #expect(maxAbsDiff(r.diagonal, f.diagonal) < 1e-4)
        let e = ExactDiagonal().estimate(s.target, x: xt, logSNR: 5, conditions: c, key: key)
        let closed = s.target.exactDiagHessian(xt, logSNR: MLXArray([Float](repeating: 5, count: 4)), conditions: c)
        let sigma = s.target.scales(5).sigma
        #expect(maxAbsDiff(e.diagonal, -sigma * closed) / max(abs(closed).max().item(Float.self) * sigma, 1e-6) < 1e-3)
    }

    /// Probe vectors are drawn at the full batch shape, then sliced: chunking changes nothing.
    @Test func chunkingNeverChangesEstimates() {
        let s = F.scenario(kappa: 1)
        let (xt, _) = noised(s, s.members[0], 4, draws: 7)
        let key = ProbeKey(schedule: F.schedule, level: 5)
        let whole = HutchinsonReverse(probes: 3).estimate(s.target, x: xt, logSNR: 4, conditions: c, key: key)
        let chunked = HutchinsonReverse(probes: 3).estimate(ChunkedDenoiser(s.target, maxBatch: 3), x: xt, logSNR: 4,
                                                            conditions: c, key: key)
        #expect(maxAbsDiff(whole.diagonal, chunked.diagonal) < 1e-6)
        #expect(maxAbsDiff(whole.epsilon, chunked.epsilon) < 1e-6)
    }
}

struct ScheduleTests {
    let key = SeedSchedule(keyData: Data("test-key-1".utf8))

    @Test func streamsAreDeterministicPerKeyAndIndependent() {
        let again = SeedSchedule(keyData: Data("test-key-1".utf8))
        #expect(key.value(.noise, 5) == again.value(.noise, 5))
        #expect(key.value(.noise, 5) != SeedSchedule(keyData: Data("test-key-2".utf8)).value(.noise, 5))
        #expect(key.value(.noise, 5) != key.value(.confirmation, 5), "detection and confirmation draws are independent")
        #expect(key.value(.scout, 5) != key.value(.noise, 5), "the scout never sees the measured draws")
        #expect(SeedSchedule.levelKey(4) == SeedSchedule.levelKey(4.0) && SeedSchedule.levelKey(4) != SeedSchedule.levelKey(5))
    }

    @Test func scheduleIDNeverRevealsTheKey() throws {
        let secret = Data("a-very-secret-deployment-key".utf8)
        let s = SeedSchedule(keyData: secret)
        #expect(s.info.id.count == 16)
        #expect(!s.info.id.contains("secret"))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scorpion-key-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let fromEnv = try SeedSchedule.load(environment: ["SCORPION_SEED_KEY": "00ff10"], directory: dir)
        #expect(fromEnv.info == SeedSchedule(keyData: Data([0x00, 0xff, 0x10])).info)
        let created = try SeedSchedule.load(environment: [:], directory: dir)
        #expect(created.info == (try SeedSchedule.load(environment: [:], directory: dir)).info)
        let perms = try FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent("seed.key").path)[.posixPermissions] as? Int
        #expect(perms == 0o600)
    }

    @Test func promptResolution() {
        #expect(ProbePrompt.resolve(user: "a dog", triggers: ["ohwx"]).text == "a dog")
        #expect(ProbePrompt.resolve(user: "", triggers: ["ohwx"]) == ProbePrompt(text: "ohwx", source: "trigger"))
        #expect(!ProbePrompt.resolve(user: nil, triggers: []).isConditional)
    }

    @Test func contentIDIsStable() {
        #expect(Fixtures.image(seed: 1).id == Fixtures.image(seed: 1).id)
        #expect(Fixtures.image(seed: 1).id != Fixtures.image(seed: 2).id)
    }

    @Test func statistics() {
        #expect(Statistics.auc(scores: [3, 2, 1, 0], labels: [true, true, false, false]) == 1)
        #expect(Statistics.exceedanceP(5, controls: [1, 2, 3]) == 0.25)
        #expect(Statistics.exceedanceP(2, controls: [1, 2, 3]) == 0.75)
        #expect(Statistics.t95(df: 7) == 1.895)
        #expect(abs(Statistics.spearman([1, 2, 3], [10, 20, 30]) - 1) < 1e-12)
    }
}
