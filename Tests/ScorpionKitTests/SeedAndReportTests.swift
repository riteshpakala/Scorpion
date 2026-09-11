import CoreGraphics
import Foundation
import MLX
import Testing
@testable import ScorpionKit

struct SeedEngineTests {
    let descriptors = [
        Descriptor(category: "subject", phrase: "a woman", probability: 0.9, source: "full"),
        Descriptor(category: "hair color", phrase: "red hair", probability: 0.7, source: "face0"),
        Descriptor(category: "eyewear", phrase: "glasses", probability: 0.6, source: "face0"),
    ]

    let key = SeedSchedule(keyData: Data("test-key-1".utf8))

    @Test func scheduleIsDeterministicPerKeyAndSharedAcrossReferences() {
        let a = SeedEngine.plan(schedule: key, descriptors: descriptors, count: 16)
        let b = SeedEngine.plan(schedule: SeedSchedule(keyData: Data("test-key-1".utf8)), descriptors: descriptors, count: 16)
        #expect(a.permutations == b.permutations)
        // Noise and λ don't depend on the reference's content — only prompts do.
        let other = SeedEngine.plan(schedule: key, descriptors: [descriptors[0]], count: 16)
        #expect(a.permutations.map(\.noiseKey) == other.permutations.map(\.noiseKey))
        #expect(a.permutations.map(\.logSNR) == other.permutations.map(\.logSNR))
    }

    @Test func differentKeysGiveDifferentStreams() {
        let a = SeedEngine.plan(schedule: key, descriptors: descriptors, count: 8)
        let c = SeedEngine.plan(schedule: SeedSchedule(keyData: Data("test-key-2".utf8)), descriptors: descriptors, count: 8)
        #expect(a.permutations.map(\.noiseKey) != c.permutations.map(\.noiseKey))
        #expect(key.value(.noise, 0) != key.value(.needle, 0), "streams are independent")
    }

    @Test func scheduleIDNeverRevealsTheKey() throws {
        let secret = Data("a-very-secret-deployment-key".utf8)
        let s = SeedSchedule(keyData: secret)
        #expect(s.info.id.count == 16)
        #expect(!s.info.id.contains("secret"))
        #expect(s.info.version == "scorpion.seed.v2")
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scorpion-key-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let fromEnv = try SeedSchedule.load(environment: ["SCORPION_SEED_KEY": "00ff10"], directory: dir)
        #expect(fromEnv.info == SeedSchedule(keyData: Data([0x00, 0xff, 0x10])).info)
        let created = try SeedSchedule.load(environment: [:], directory: dir)
        let reloaded = try SeedSchedule.load(environment: [:], directory: dir)
        #expect(created.info == reloaded.info)
        let perms = try FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent("seed.key").path)[.posixPermissions] as? Int
        #expect(perms == 0o600)
    }

    @Test func triggersEnterPromptsAndPromptSet() {
        let plan = SeedEngine.plan(schedule: key, descriptors: descriptors, count: 8, triggers: ["ohwx woman"])
        #expect(plan.permutations.filter { $0.prompt.hasPrefix("ohwx woman, ") }.count == 4)
        let set = PromptSet.make(descriptors: descriptors, triggers: ["ohwx woman"])
        #expect(set.attacker.hasPrefix("ohwx woman, a photo of a woman"))
        #expect(PromptSet.make(descriptors: descriptors, triggers: []).attacker == set.descriptor)
    }

    @Test func logSNRIsStratifiedOverTheBand() {
        let band = LogSNRBand(lo: -4, hi: 2)
        let plan = SeedEngine.plan(schedule: key, descriptors: descriptors, count: 32, band: band)
        let strata = plan.permutations.map { Int((($0.logSNR - band.lo) / band.width) * 32) }
        #expect(Set(strata) == Set(0..<32), "one sample per stratum")
    }

    @Test func promptsUseDescriptorsAndNullsSwapThem() {
        let plan = SeedEngine.plan(schedule: key, descriptors: descriptors, count: 24)
        for p in plan.permutations {
            #expect(p.prompt.contains("a woman"))
            #expect(p.prompt != p.nullPrompt)
            #expect(!p.nullPrompt.contains("red hair") || !p.nullPrompt.contains("glasses") || !p.prompt.contains("red hair"))
        }
    }

    @Test func noiseIsIdenticalForEveryBackend() {
        let plan = SeedEngine.plan(schedule: key, descriptors: [], count: 2)
        let a = SeedPlan.noise(plan.permutations[0], shape: [1, 4, 4])
        let b = SeedPlan.noise(plan.permutations[0], shape: [1, 4, 4])
        let c = SeedPlan.noise(plan.permutations[1], shape: [1, 4, 4])
        #expect(maxAbsDiff(a, b) == 0)
        #expect(maxAbsDiff(a, c) > 0)
    }

    @Test func bandParsing() {
        #expect(LogSNRBand(parsing: "-4, 2") == LogSNRBand(lo: -4, hi: 2))
        #expect(LogSNRBand(parsing: "2,-4") == LogSNRBand(lo: -4, hi: 2))
        #expect(LogSNRBand(parsing: "nope") == nil)
    }
}

struct ReferenceTests {
    @Test func contentIDAndStats() {
        let a = Fixtures.image(seed: 1), b = Fixtures.image(seed: 1), c = Fixtures.image(seed: 2)
        #expect(a.id == b.id && a.id != c.id)
        let stats = VisualStats.compute(a)
        #expect(stats.perceptualHash.count == 16)
        #expect(stats.luminanceEntropyBits > 3)
        #expect(stats.spectralSlope < 0)
        #expect(VisualStats.histogramEntropy([Float](repeating: 0.5, count: 100)) == 0)
    }

    @Test func maskGridFollowsCropAndFace() {
        let face = FaceRegion(index: 0, bounds: CGRect(x: 40, y: 40, width: 40, height: 40), confidence: 1,
                              landmarks: ["nose": CGPoint(x: 60, y: 62)])
        let mask = RegionMask.face(face)
        #expect(mask.weight(x: 60, y: 58) > 0.9)
        #expect(mask.weight(x: 5, y: 5) < 0.01)
        let grid = mask.grid(crop: CGRect(x: 0, y: 0, width: 128, height: 128), width: 8, height: 8)
        #expect(grid[3 * 8 + 3] > 0.5 && grid[0] < 0.01)
    }

    @Test func cropsStayInsideTheImage() {
        let img = Fixtures.image(width: 200, height: 120)
        let face = FaceRegion(index: 0, bounds: CGRect(x: 170, y: 5, width: 25, height: 30), confidence: 1, landmarks: [:])
        for crop in CropPlanner.crops(for: img, faces: [face]) {
            #expect(crop.rect.minX >= 0 && crop.rect.minY >= 0)
            #expect(crop.rect.maxX <= 200.001 && crop.rect.maxY <= 120.001)
            #expect(abs(crop.rect.width - crop.rect.height) < 0.001)
        }
    }

    @Test func clipTokenizerBPE() throws {
        let vocab = #"{"<|startoftext|>": 0, "<|endoftext|>": 1, "h": 2, "i</w>": 3, "hi</w>": 4, "a</w>": 5}"#
        let tok = try CLIPTokenizer(vocabJSON: Data(vocab.utf8), merges: "#version: 0.2\nh i</w>\n")
        #expect(tok.encode("Hi  a") == [0, 4, 5, 1])
        #expect(tok.encode("") == [0, 1])
    }
}

struct ScoreAndReportTests {
    @Test func provisionalFusionIsWeightedLogitMean() {
        let comps = [ScoreComponent(kind: .metadataMatch, value: 0.8, weight: 1, note: ""),
                     ScoreComponent(kind: .personAdapter, value: 0.8, weight: 3, note: "")]
        #expect(abs(ScoreFusion.fuse(comps, calibration: nil) - 0.8) < 1e-9)
        #expect(ScoreFusion.fuse([], calibration: nil) == 0.5)
    }

    @Test func calibrationSeparatesLabeledData() {
        var samples: [(components: [ScoreComponent], label: Bool)] = []
        for i in 0..<40 {
            let pos = i % 2 == 0
            let v = pos ? 0.7 + Double(i % 5) * 0.05 : 0.1 + Double(i % 5) * 0.05
            samples.append(([ScoreComponent(kind: .metadataMatch, value: v, weight: 1, note: "")], pos))
        }
        let cal = ScoreFusion.fit(samples)
        #expect(cal.auc == 1)
        #expect(cal.weights[ScoreComponent.Kind.metadataMatch.rawValue]! > 0)
        #expect(ScoreFusion.auc(scores: [0.1, 0.9], labels: [false, true]) == 1)
    }

    @Test func metadataMatcherLexical() {
        var f = MetadataFindings()
        f.trainingTags = [TagCount(tag: "red hair", count: 10), TagCount(tag: "a woman with glasses on a beach", count: 4)]
        f.trainingTagTotal = 14
        let d = [Descriptor(category: "hair color", phrase: "red hair", probability: 0.8, source: "face0"),
                 Descriptor(category: "eyewear", phrase: "glasses", probability: 0.6, source: "face0"),
                 Descriptor(category: "facial hair", phrase: "a beard", probability: 0.5, source: "face0")]
        let m = MetadataMatcher.match(f, descriptors: d, clip: nil)!
        #expect(m.matches.first?.descriptor == "red hair" && m.matches.first?.similarity == 1)
        #expect(m.matches.contains { $0.descriptor == "glasses" })
        #expect(!m.matches.contains { $0.descriptor == "a beard" })
        #expect(m.score > 0.3 && m.score < 1)
    }

    @Test func reportRoundTrips() throws {
        let plan = SeedEngine.plan(schedule: .research, descriptors: [], count: 2)
        let report = LikenessReport(
            schema: LikenessReport.schema, reportID: UUID(), timestamp: Date(timeIntervalSince1970: 1_800_000_000),
            scorpionVersion: ScorpionInfo.version,
            reference: ReferenceSummary(referenceID: "r", name: "x.png", width: 8, height: 8, faces: [], selectedFace: nil,
                                        descriptors: [], visualStats: VisualStats.compute(Fixtures.image()),
                                        seedSchedule: plan.schedule, prompts: PromptSet(descriptor: "a photo", triggers: []), segmentation: [],
                                        permutations: 2, band: .identity, clipModel: nil),
            source: SourceSummary(link: "https://huggingface.co/o/r", host: .huggingFace, repo: "o/r", commit: "abc",
                                  files: [], repoBytes: 10, weightBytes: 8, bytesFetched: 2, bytesFromCache: 0, requests: 1),
            units: [], flags: ["pickle"], metadata: MetadataFindings(), metadataMatch: nil, bankNeighbors: [],
            probes: ProbeSection(likelihood: nil, note: "n/a"),
            components: [ScoreComponent(kind: .personAdapter, value: 0.25, weight: 0.75, note: "")],
            harmScore: 0.25, harmScoreCI90: nil, calibrated: false, disclaimer: LikenessReport.disclaimer)
        let back = try LikenessReport.decode(try report.jsonData())
        #expect(back.schema == "scorpion.report.v2")
        #expect(back.reference.seedSchedule == SeedSchedule.research.info)
        #expect(back.source.fetchedFraction == 0.25)
        #expect(back.components.first?.kind == .personAdapter)
    }
}

@Suite(.serialized) struct FingerprintBankTests {
    @Test func addQueryAndPersist() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scorpion-bank-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let v1 = MLXArray(Fixtures.randomFloats(16, seed: 1), [16]), v2 = MLXArray(Fixtures.randomFloats(16, seed: 2), [16])
        let bank = try FingerprintBank(directory: dir)
        try bank.add(label: "Identity", familyKey: "fam", source: "a@1", unit: "u", fingerprints: ["m1": v1, "m2": v2])
        try bank.add(label: "style", familyKey: "fam", source: "b@1", unit: "u", fingerprints: ["m1": v2, "m2": v1])
        try bank.add(label: "identity", familyKey: "other", source: "c@1", unit: "u", fingerprints: ["m1": v1, "m2": v2])

        let reloaded = try FingerprintBank(directory: dir)
        #expect(reloaded.entries.count == 3)
        let hits = reloaded.topK(["m1": -v1, "m2": v2], familyKey: "fam")
        #expect(hits.count == 2)
        #expect(hits.first?.label == "identity" && abs(hits.first!.similarity - 1) < 1e-5, "sign-invariant match")
        let vote = FingerprintBank.likenessVote(hits)!
        #expect(vote > 0.5 && vote < 1)
    }
}
