import Foundation
import Testing
@testable import ScorpionKit

@Suite(.serialized) struct ExperimentTests {
    @Test func toyExperimentPassesTheProtocol() throws {
        var toy = ExperimentManifest.Toy()
        toy.identities = 2
        toy.memorization.kappas = [0, 0.5, 1]
        toy.memorization.references = 3
        let report = try ExperimentRunner.run(ExperimentManifest(name: "test", toy: toy), schedule: ToyFixtures.schedule)
        #expect(report.rows.count == 2 * 3 * 2 * 3)
        #expect(report.criteria.map(\.name) == ["membership", "localization", "dose-response", "specificity"])
        #expect(report.go, "\(report.criteria.map { "\($0.name) \($0.value)" })")
    }

    @Test func verdictFailsWithoutMembershipSignal() {
        func row(_ label: ExperimentManifest.Label, _ kappa: Double, _ collapse: Double, _ snap: Double) -> ExperimentReport.Row {
            .init(pair: "p", label: label, kappa: kappa, collapse: collapse, snap: snap, iou: 1,
                  verdict: collapse > 0 ? .memorized : .noEvidence, denoiserCalls: 1)
        }
        let rows = [row(.member, 0, 0, 1), row(.heldOut, 0, 0, 1), row(.member, 1, 0.7, 1.1), row(.member, 1, 0.7, 0.9),
                    row(.heldOut, 1, 0, 1.2), row(.heldOut, 1, 0, 1.0)]
        let (criteria, go) = ExperimentRunner.verdict(rows)
        #expect(!go)
        #expect(criteria.first { $0.name == "membership" }?.pass == false)
        #expect(criteria.first { $0.name == "dose-response" }?.pass == true)
    }

    @Test func exampleManifestDecodes() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Docs/examples/toy-memory.json")
        let manifest = try JSONDecoder().decode(ExperimentManifest.self, from: Data(contentsOf: url))
        #expect(manifest.toy?.memorization.kappas == [0, 0.5, 0.9, 1])
    }
}
