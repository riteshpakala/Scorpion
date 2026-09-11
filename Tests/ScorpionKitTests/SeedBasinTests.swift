//
//  The optional seed-endpoint basin, on top of a core result.
//

import Foundation
import MLX
import Testing
@testable import ScorpionKit

/// Ground truth (toy only): the sample is the subject inside the region.
struct IdentityOracle: SeedEndpointCriterion {
    let scenario: MemoryScenario
    let threshold = 0.0
    func scores(_ x0: MLXArray) -> [Double] { scenario.identityMargin(x0).map(Double.init) }
}

@Suite(.serialized) struct SeedBasinTests {
    typealias F = ToyFixtures

    /// A stored region is an attractor: perturbed seeds return to it under the target, not the base.
    @Test func storedRegionIsAWiderAttractor() throws {
        let s = F.scenario(kappa: 1, identityWeight: 0.3)
        let ref = F.reference(s, s.members[0])
        let models = ModelPair(target: s.target, base: s.base)
        let result = try MemorizationProbe(configuration: .toy).run(models: models, reference: ref, prompt: F.prompt, schedule: F.schedule)
        #expect(result.verdict == .memorized)
        var config = SeedBasinConfiguration()
        config.trials = 24
        config.sampler = DeterministicSampler(solver: .ddim, steps: 96)
        // "Distinct images" scale: other photos of the subject (the photo-to-photo spread).
        let samples = ([s.members[0]] + s.photos(of: s.identityMean, count: 4, schedule: F.schedule, index: 2))
            .map { MLXArray($0, s.latentShape) }
        let b = try #require(try SeedBasinTest(configuration: config).run(models: models, reference: ref, result: result,
                                                                          scaleSamples: samples, schedule: F.schedule))
        #expect(b.region == result.regions[0].id)
        #expect(b.halfRadiusTarget > b.halfRadiusBase + 0.2, "β½ \(b.halfRadiusTarget) vs base \(b.halfRadiusBase)")
        #expect(zip(b.target, b.base).allSatisfy { $0 + 0.1 >= $1 }, "\(b.target) vs \(b.base)")
        #expect(b.supportsMemorization)
    }

    /// r(1) is the chance a random seed lands in the region: it matches brute-force sampling.
    @Test func fullPerturbationIsTheRegionMass() throws {
        let s = F.scenario(kappa: 1, identityWeight: 0.3)
        let ref = F.reference(s, s.members[0])
        var config = SeedBasinConfiguration()
        config.trials = 64
        config.betas = [0, 1]
        config.sampler = DeterministicSampler(solver: .ddim, steps: 96)
        let b = try SeedBasinTest(configuration: config).run(
            models: ModelPair(target: s.target, base: s.base), reference: ref, regionMask: s.memoryMask, regionID: 1,
            criterion: IdentityOracle(scenario: s), prompt: F.prompt, schedule: F.schedule)
        let z = F.schedule.normal(.jitter, 50_000, shape: [1000] + s.latentShape)
        let x = config.sampler.sample(s.target, noise: z, conditions: [Conditioning(prompt: F.prompt.text)])
        let brute = Double(s.identityMargin(x).filter { $0 > 0 }.count) / 1000
        let r1 = b.target.last!
        #expect(abs(r1 - brute) <= 3 * (brute * (1 - brute) / 64).squareRoot() + 0.02, "r(1) \(r1) vs μ \(brute)")
    }

    @Test func noRegionNoBasin() throws {
        let s = F.scenario(kappa: 0)
        let ref = F.reference(s, s.members[0])
        let models = ModelPair(target: s.target, base: s.base)
        let result = try MemorizationProbe(configuration: .toy).run(models: models, reference: ref, prompt: F.prompt, schedule: F.schedule)
        let samples = s.members.prefix(3).map { MLXArray($0, s.latentShape) }
        #expect(try SeedBasinTest().run(models: models, reference: ref, result: result, scaleSamples: samples,
                                        schedule: F.schedule) == nil)
    }

    @Test func composeKeepsSeedsGaussian() {
        let a = MLXArray.ones([4]), b = MLXArray.zeros([4])
        let m = MLXArray([Float(1), 0, 0.6, 0.8])
        let z = SeedComposer.compose(a, b, footprint: m).asArray(Float.self)
        #expect(z == [1, 0, 0.6, 0.8])
        #expect(SeedBasinTest.halfRadius([0, 0.5, 1], [1, 0.5, 0]) == 0.5)
        #expect(SeedBasinTest.halfRadius([0, 0.5, 1], [0.4, 0.2, 0]) == 0)
        #expect(SeedBasinTest.halfRadius([0, 0.5, 1], [1, 1, 1]) == 1)
    }
}
