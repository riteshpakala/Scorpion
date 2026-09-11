import Foundation
import Testing
@testable import ScorpionKit

@Suite(.serialized) struct ExperimentTests {
    @Test func toyExperimentScoresEveryPairAndRendersAVerdict() throws {
        var toy = ExperimentManifest.Toy()
        toy.identities = 1
        toy.lookAlikesPerIdentity = 1
        toy.unrelated = 1
        toy.styleModel = false
        var settings = ExperimentManifest.Settings()
        settings.samplesPerLevel = 60
        settings.maxLevels = 3
        settings.steps = 8
        settings.faceProbePermutations = 32
        let report = try ExperimentRunner.run(ExperimentManifest(name: "t", toy: toy, settings: settings, pairs: nil))
        #expect(report.rows.count == 3)
        #expect(report.criteria.map(\.name) == ["discrimination", "locality", "precision & cost", "early exit"])
        let match = report.rows.first { $0.label == .identityMatch }!
        let alike = report.rows.first { $0.label == .lookAlike }!
        #expect(match.faceDeltaBits > alike.faceDeltaBits + 5, "match \(match.faceDeltaBits) vs look-alike \(alike.faceDeltaBits)")
        #expect(report.auc["face: match vs look-alike"] == 1)
        #expect(report.seedSchedule == SeedSchedule.research.info)
    }

    @Test func realBackendsNeedAnExecutor() {
        let pair = ExperimentManifest.Pair(id: "p", backend: "flux2-klein:lora", base: "flux2-klein:base",
                                           references: [], controls: [], label: .identityMatch)
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentRunner.run(ExperimentManifest(name: "x", toy: nil, settings: nil, pairs: [pair]))
        }
    }

    @Test func verdictFailsWhenLookAlikesScoreLikeMatches() {
        func row(_ label: ExperimentManifest.Label, _ bits: Double) -> ExperimentReport.Row {
            .init(pair: "\(label)-\(bits)", label: label, needleDeltaBits: bits, needleIsLowerBound: false, faceDeltaBits: bits,
                  backgroundIndependence: 0.9, footprintFraction: 0.2, ciHalfWidthBits: 1, earlyExitAgreement: 0.99,
                  denoiserCalls: 1000, seconds: 0)
        }
        let report = ExperimentRunner.verdict(name: "v", rows: [row(.identityMatch, 3), row(.identityMatch, 1),
                                                                row(.lookAlike, 2), row(.lookAlike, 4)],
                                              schedule: .research)
        #expect(!report.go)
        #expect(report.criteria.first { $0.name == "discrimination" }!.pass == false)
        #expect(report.criteria.first { $0.name == "locality" }!.pass)
    }

    @Test func exampleManifestsDecode() throws {
        let docs = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Docs/examples")
        let manifest = try JSONDecoder().decode(ExperimentManifest.self,
                                                from: Data(contentsOf: docs.appendingPathComponent("toy-experiment.json")))
        #expect(manifest.toy?.identities == 3)
        #expect(manifest.settings?.faceBand == LogSNRBand(lo: -6, hi: 8))
        #expect(manifest.toy?.memorization == nil, "older manifests keep decoding")
        let memory = try JSONDecoder().decode(ExperimentManifest.self,
                                              from: Data(contentsOf: docs.appendingPathComponent("toy-memory.json")))
        #expect(memory.toy?.memorization?.kappas == [0, 0.5, 0.9, 1])
        #expect(memory.toy?.needleArm == false)
    }

    /// Tier M on the toy: members separate from held-out photos once photos are stored, the
    /// pool sits on the face, it grows with memorization, and learned-only faces aren't flagged.
    @Test func memoryArmRendersTierMVerdict() throws {
        var toy = ExperimentManifest.Toy()
        toy.identities = 1
        toy.needleArm = false
        var memorization = ExperimentManifest.Memorization()
        memorization.kappas = [0, 0.5, 1]
        memorization.members = 6
        memorization.references = 3
        toy.memorization = memorization
        let report = try ExperimentRunner.run(ExperimentManifest(name: "m", toy: toy, settings: nil, pairs: nil))
        #expect(report.rows.isEmpty && report.criteria.isEmpty)
        let rows = report.memoryRows!
        #expect(rows.count == 3 * 2 * 3)
        #expect(report.memoryCriteria!.map(\.name) == ["membership", "localization", "dose-response", "specificity"])
        for c in report.memoryCriteria! { #expect(c.pass, "\(c.name): \(c.value)") }
        #expect(report.memoryGo == true)
    }

    @Test func memoryVerdictFailsWithoutMembershipSignal() {
        func row(_ label: ExperimentManifest.Label, _ kappa: Double, _ collapse: Double, _ snap: Double) -> ExperimentReport.MemoryRow {
            .init(pair: "p", label: label, kappa: kappa, collapse: collapse, snap: snap, iou: 0.9, tier: .noEvidence, denoiserCalls: 1)
        }
        let rows = [row(.member, 0, 0, 1), row(.heldOut, 0, 0, 1), row(.member, 1, 0.8, 2), row(.member, 1, 0.7, 1),
                    row(.heldOut, 1, 0.1, 2), row(.heldOut, 1, 0.2, 1)]
        let (criteria, go) = ExperimentRunner.memoryVerdict(rows)
        #expect(!go)
        #expect(criteria.first { $0.name == "membership" }!.pass == false)
        #expect(criteria.first { $0.name == "dose-response" }!.pass)
    }
}
