//
//  The memorization probe against toy ground truth: known planted regions, known
//  memorization strengths κ, known training members.
//

import CoreGraphics
import Foundation
import MLX
import Testing
@testable import ScorpionKit

@Suite(.serialized) struct MemorizationProbeTests {
    typealias F = ToyFixtures

    @Test func storedRegionLocalizesAndIsMemorized() throws {
        let s = F.scenario(kappa: 1)
        let r = try F.run(s)
        #expect(r.verdict == .memorized, "\(r.verdict): \(r.reasons)")
        #expect(r.regions.count == 1)
        #expect(F.iou(r.regions.first, F.planted(s)) >= 0.9, "IoU \(F.iou(r.regions.first, F.planted(s)))")
        #expect(r.regions[0].meanCollapse >= 0.6, "mean collapse \(r.regions[0].meanCollapse)")
        #expect(r.regions[0].snap >= 4)
        #expect(r.regions[0].significance == .uncalibrated, "no controls → uncalibrated")
        // Nothing lights up outside the planted region.
        let raw = r.heatmap.layer("collapse-raw")!.values
        let outside = F.planted(s).indices.filter { !F.planted(s)[$0] }.map { raw[$0] }
        #expect(outside.max()! < 0.1, "background collapse \(outside.max()!)")
        // The primary layer is zero outside counted regions.
        let counted = Set(r.regions.filter(\.counts).flatMap(\.cells))
        let primary = r.heatmap.primary!.values
        #expect(primary.indices.allSatisfy { counted.contains($0) || primary[$0] == 0 })
    }

    /// The estimate tracks the closed-form collapse over the planted region.
    @Test func collapseFieldMatchesExactCollapse() throws {
        let s = F.scenario(kappa: 0.9)
        let ref = F.reference(s, s.members[0])
        let models = ModelPair(target: s.target, base: s.base)
        let levels: [Float] = [3, 4, 5]
        let field = try CollapseFieldStage(estimator: HutchinsonReverse(probes: 16), draws: 8)
            .measure(models: models, reference: ref, prompt: F.prompt, levels: levels, adjacency: [true, true], schedule: F.schedule)
        let truth = F.planted(s)
        for (l, lambda) in levels.enumerated() {
            let (xt, _, _) = CollapseFieldStage.noised(s.target, x0: ref.latent, logSNR: lambda, draws: 8, schedule: F.schedule)
            let lv = MLXArray([Float](repeating: lambda, count: 8)), c = [Conditioning(prompt: F.prompt.text)]
            let exact = GridMath.mean(GridMath.rows(s.base.exactRetainedVariance(xt, logSNR: lv, conditions: c)
                - s.target.exactRetainedVariance(xt, logSNR: lv, conditions: c), grid: ref.grid))
            let est = GridMath.mean(field.collapse[l])
            let cells = truth.indices.filter { truth[$0] }
            let e = cells.reduce(0.0) { $0 + Double(est[$1]) } / Double(cells.count)
            let x = cells.reduce(0.0) { $0 + Double(exact[$1]) } / Double(cells.count)
            #expect(abs(e - x) < 0.03, "λ = \(lambda): estimated region collapse \(e) vs exact \(x)")
        }
    }

    /// The heatmap's intensity is the memorization strength: stored > half-memorized > learned.
    @Test func intensityRanksByMemorizationStrength() throws {
        let side = 16
        let plants: [PlantedRegion] = [
            .rect("stored", x: 0, y: 0, width: 0.4, height: 0.4, side: side, kappa: 1),
            .rect("half", x: 0.55, y: 0.05, width: 0.4, height: 0.4, side: side, kappa: 0.6),
            .rect("learned", x: 0.3, y: 0.55, width: 0.4, height: 0.4, side: side, kappa: 0),
        ]
        let s = F.scenario(kappa: 1, regions: plants, side: side)
        let r = try F.run(s)
        func best(_ i: Int) -> MemorizedRegion? { r.regions.max { F.iou($0, F.planted(s, i)) < F.iou($1, F.planted(s, i)) } }
        let stored = best(0), half = best(1)
        #expect(F.iou(stored, F.planted(s, 0)) >= 0.7 && F.iou(half, F.planted(s, 1)) >= 0.5,
                "IoU stored \(F.iou(stored, F.planted(s, 0))), half \(F.iou(half, F.planted(s, 1)))")
        #expect((stored?.meanCollapse ?? 0) > (half?.meanCollapse ?? 0) + 0.1)
        #expect(F.iou(best(2), F.planted(s, 2)) < 0.2, "the learned region is not memorized")
        #expect(r.regions.first.map { F.iou($0, F.planted(s, 0)) >= 0.7 } == true, "strongest region first")
    }

    @Test func collapseGrowsWithMemorization() throws {
        let peaks = try [0.0, 0.5, 0.9, 1.0].map { k -> Double in
            try F.run(F.scenario(kappa: k)).regions.first?.meanCollapse ?? 0
        }
        #expect(zip(peaks, peaks.dropFirst()).allSatisfy { $0 < $1 }, "\(peaks)")
        #expect(peaks[0] == 0, "κ = 0 (learned) leaves no region")
    }

    /// Common random numbers: a model compared with itself reads exactly zero.
    @Test func identicalModelsGiveExactlyZero() throws {
        let s = F.scenario(kappa: 1)
        let ref = F.reference(s, s.members[0])
        let r = try MemorizationProbe(configuration: .toy).run(models: ModelPair(target: s.base, base: s.base), reference: ref,
                                                               prompt: F.prompt, schedule: F.schedule)
        #expect(r.regions.isEmpty)
        #expect(r.verdict == .noEvidence)
        for layer in r.heatmap.layers { #expect(layer.values.allSatisfy { $0 == 0 }, "\(layer.name)") }
        #expect(r.levels.allSatisfy { $0.snap == 1 })
    }

    @Test(arguments: [0.5, 1.0])
    func heldOutReferenceIsNeverMemorized(kappa: Double) throws {
        let s = F.scenario(kappa: kappa, member: false)
        let r = try F.run(s, reference: s.heldOut[0])
        #expect(r.verdict != .memorized, "\(r.verdict): \(r.reasons)")
    }

    /// Conditional overfitting: under a trigger the conditional model stores the photos while
    /// the unconditional one keeps only a smoothed subject — the prompt pins the region.
    @Test func promptPinsTheMemoryWhenTheUnconditionalIsUnderfitted() throws {
        let gated = F.scenario(kappa: 1, trigger: "ohwx")
        let r = try F.run(gated, prompt: ProbePrompt(text: "ohwx", source: "trigger"))
        #expect((r.regions.first?.promptCollapse ?? 0) >= 0.3, "prompt collapse \(r.regions.first?.promptCollapse ?? 0)")
        #expect(r.heatmap.layer("prompt-collapse") != nil)
        let ungated = F.scenario(kappa: 1)
        let u = try F.run(ungated)
        #expect(abs(u.regions.first?.promptCollapse ?? 1) < 0.02, "same model for every prompt: nothing to pin")
        let unconditional = try F.run(ungated, prompt: ProbePrompt.resolve(user: nil, triggers: []))
        #expect(unconditional.heatmap.layer("prompt-collapse") == nil)
    }

    /// Confound: for content the base has never seen, the base hesitates between contents at
    /// high noise — extra variance on its side that reads as "collapse" for any model that
    /// merely learned it. Release gating per position removes it.
    @Test func learnedNovelContentIsNotFlagged() throws {
        let side = F.side
        let novel = (0..<(side * side)).map { i -> Float in ((i / side) + (i % side)) % 2 == 0 ? 1.5 : -1.5 }
        let s = F.scenario(kappa: 0, pixels: novel)
        let r = try F.run(s, reference: novel)
        #expect(r.regions.isEmpty, "learned novel content: \(r.regions.map(\.meanCollapse))")
        let field = try CollapseFieldStage(estimator: HutchinsonReverse(probes: 8), draws: 8).measure(
            models: ModelPair(target: s.target, base: s.base), reference: F.reference(s, novel), prompt: F.prompt,
            levels: [-2, 0, 1, 2, 3], adjacency: [true, true, true, true], schedule: F.schedule)
        let raw = field.collapse.map { GridMath.mean($0).max() ?? 0 }.max() ?? 0
        #expect(raw > 0.1, "the confound is real: ungated single-level collapse \(raw)")
        #expect(try F.run(F.scenario(kappa: 1, pixels: novel), reference: novel).verdict == .memorized)
    }

    @Test func wholePhotoMemoryIsNotLocal() throws {
        let s = F.scenario(kappa: 1, regions: [.whole(side: F.side, kappa: 1)])
        let r = try F.run(s)
        #expect(r.verdict == .memorized)
        #expect(r.regions.reduce(0) { $0 + $1.areaFraction } >= 0.7, "area \(r.regions.map(\.areaFraction))")
    }

    /// One photo is enough for membership — when there is something photo-specific to find.
    @Test func membershipAUCOnlyWhenMemorized() throws {
        var config = MemorizationConfiguration.toy
        config.probes = 4
        config.scoutLevels = [2, 3, 4, 5, 6, 7, 8]
        for kappa in [1.0, 0.0] {
            let s = F.scenario(kappa: kappa)
            let members = try s.members.prefix(5).map { try F.run(s, reference: $0, config: config).membershipSnap }
            let held = try s.heldOut.prefix(5).map { try F.run(s, reference: $0, config: config).membershipSnap }
            let auc = Statistics.auc(scores: members + held, labels: members.map { _ in true } + held.map { _ in false })!
            if kappa == 1 {
                #expect(auc >= 0.9, "κ = 1 membership AUC \(auc)")
            } else {
                #expect(abs(auc - 0.5) <= 0.3, "κ = 0 membership AUC \(auc) — nothing photo-specific is stored")
            }
        }
    }
}

@Suite(.serialized) struct NullCalibrationTests {
    typealias F = ToyFixtures

    static var config: MemorizationConfiguration {
        var c = MemorizationConfiguration.toy
        c.probes = 4
        c.scoutLevels = [2, 3, 4, 5, 6, 7, 8]
        return c
    }

    /// Non-members: look-alike and stranger photos the fine-tune never saw.
    func controls(_ s: MemoryScenario, count: Int) -> [[Float]] {
        Array(s.controlPhotos(perLookAlike: 3, perCrowd: 2, schedule: F.schedule).prefix(count))
    }

    @Test func storedRegionBeatsNineteenControls() throws {
        let s = F.scenario(kappa: 1)
        var c = Self.config
        c.minimumControls = 19
        let r = try F.run(s, config: c, controls: controls(s, count: 19))
        #expect(r.controls?.count == 19)
        #expect(r.regions.first?.significance == .significant, "p = \(r.regions.first?.pValue ?? -1)")
        #expect(abs((r.regions.first?.pValue ?? 1) - 0.05) < 1e-9, "p = 1/(1+19) when no control comes close")
        #expect(r.verdict == .memorized, "\(r.reasons)")
    }

    @Test func learnedContentStaysBelowTheControls() throws {
        let s = F.scenario(kappa: 0)
        let r = try F.run(s, config: Self.config, controls: controls(s, count: 9))
        #expect(!r.regions.contains { $0.significance == .significant })
        #expect(r.verdict == .noEvidence)
    }

    @Test func evidenceProfileNeedsItsControls() throws {
        let s = F.scenario(kappa: 1)
        var c = Self.config
        c.minimumControls = 19
        let r = try F.run(s, config: c)
        #expect(r.verdict == .inconclusive, "\(r.reasons)")
        let issues = try MemorizationProbe(configuration: Self.config).run(
            models: ModelPair(target: s.target, base: s.base, issues: ["adapter trained on another base"]),
            reference: F.reference(s, s.members[0]), prompt: F.prompt, schedule: F.schedule)
        #expect(issues.verdict == .inconclusive && issues.reasons.contains("adapter trained on another base"))
    }

    @Test func baseSampleControlsCalibrate() throws {
        let s = F.scenario(kappa: 1)
        var c = Self.config
        c.baseSampleControls = 4
        let r = try F.run(s, config: c)
        #expect(r.controls?.source == "base-samples" && r.controls?.count == 4)
        #expect(r.regions.first?.pValue == 0.2, "p = 1/(1+4)")
        #expect(r.regions.first?.significance == .notSignificant, "4 controls can't reach α = 0.05")
    }
}

@Suite(.serialized) struct MemorizationReportTests {
    @Test func toyBackendEndToEnd() async throws {
        var options = ScorpionOptions()
        options.seedSchedule = ToyFixtures.schedule
        let scorpion = Scorpion(options: options)
        let image = Fixtures.image(width: 160, height: 120, seed: 3)
        var config = MemorizationConfiguration.toy
        config.probes = 8
        let request = BackendRequest(backend: "toy", reference: image, options: ["side": ["24"]])
        let report = try await scorpion.detectMemorization(request, configuration: config)
        let r = report.result
        #expect(report.schema == "scorpion.memorization.v1")
        #expect(r.heatmap.grid == GridShape(rows: 24, cols: 24))
        #expect(r.heatmap.frame == CGRect(x: 0, y: 0, width: 160, height: 120), "the grid covers the whole image")
        #expect(r.verdict == .memorized, "\(r.reasons)")
        let plants = ToyBackendFactory.defaultPlants(side: 24)
        #expect(ToyFixtures.iou(r.regions.first, plants[0].grid.map { $0 > 0.5 }) >= 0.7)
        #expect(report.provenance["plants"]?.contains("plant-1") == true)

        // The report round-trips.
        let encoder = JSONEncoder()
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        let back = try decoder.decode(MemorizationReport.self, from: encoder.encode(report))
        #expect(back.result.verdict == r.verdict && back.result.heatmap.primary?.values == r.heatmap.primary?.values)
        #expect(back.result.regions.map(\.cells) == r.regions.map(\.cells))
    }

    @Test func unknownBackendsNameTheAvailableOnes() async {
        do {
            _ = try await BackendRegistry.shared.resolve(BackendRequest(backend: "nope", reference: Fixtures.image()))
            Issue.record("expected an error")
        } catch {
            #expect(error.localizedDescription.contains("toy"))
        }
    }
}
