//
//  MemorizationProbe.swift
//  ScorpionKit
//
//  Has this model memorized regions of this reference image? Kim & Lee (arXiv 2605.26756):
//  verbatim memorization is coordinate-wise variance collapse — the model stops letting some
//  pixels vary — visible as curvature against an underfitted baseline. The probe reads that
//  collapse at the *noised reference* (the reference is what's on trial; nothing is generated
//  for display, nothing is decoded) and lays it over the image as a heatmap.
//
//  The process, one stage per step, each its own type:
//
//    1. LevelScout          which noise levels the base leaves the reference free at (base only)
//    2. CollapseFieldStage  per-position collapse ρ_base − ρ_target (and ρ_uncond − ρ_cond)
//    3. RegionDetector      sustained, gated collapse → candidate regions with a mass each
//    4. NullCalibration     the same steps on negative controls → a p-value per region
//    5. ConfirmationStage   on independent draws: does the target return the region to the
//                           reference itself (snap)?
//    6. VerdictRule         memorized / memorized-other-content / no-evidence / inconclusive
//
//  The optional seed-endpoint basin (`SeedBasinTest`) builds on this result; this file never
//  depends on it.
//

import CoreGraphics
import Foundation
import MLX

public struct MemorizationProbe {
    public var configuration: MemorizationConfiguration

    public init(configuration: MemorizationConfiguration = .standard) { self.configuration = configuration }

    /// - Parameters:
    ///   - reference: the reference encoded by the target's encoder.
    ///   - controls: negative controls (images known not to be training members), encoded the same way.
    ///   - imageSize: the reference image's pixel size (defaults to the frame's).
    public func run(models: ModelPair, reference: EncodedReference, prompt: ProbePrompt,
                    controls: [EncodedReference] = [], imageSize: CGSize? = nil, schedule: SeedSchedule,
                    progress: ((String) -> Void)? = nil) throws -> MemorizationResult {
        let started = Date()
        let c = configuration
        let field = CollapseFieldStage(estimator: c.estimator.make(probes: c.probes), draws: c.draws)
        let detector = RegionDetector(configuration: c)
        let confirm = ConfirmationStage(draws: c.confirmationDraws)

        // 1. Levels, from the base alone.
        progress?("scouting noise levels (base model only)")
        let scout = try LevelScout(configuration: c).run(base: models.base, reference: reference, prompt: prompt, schedule: schedule)
        var evaluations = scout.evaluations
        let levels = scout.selected

        // 2–3 (and 5), the same for the reference and for every control.
        func measure(_ image: EncodedReference) throws -> (CollapseField, RegionDetection, ConfirmationField) {
            let f = try field.measure(models: models, reference: image, prompt: prompt, levels: levels,
                                      adjacency: scout.adjacency, schedule: schedule)
            let d = detector.detect(f)
            let k = try confirm.measure(models: models, reference: image, prompt: prompt, levels: levels, schedule: schedule)
            evaluations += f.evaluations + k.evaluations
            return (f, d, k)
        }
        guard !levels.isEmpty else {
            return empty(models: models, reference: reference, prompt: prompt, scout: scout, imageSize: imageSize,
                         estimator: field.estimator.name, schedule: schedule, evaluations: evaluations, started: started)
        }
        progress?("measuring collapse at λ ∈ {\(levels.map { String(format: "%g", $0) }.joined(separator: ", "))}")
        let (f, detection, confirmation) = try measure(reference)

        // 4. Null calibration.
        var controlSummary: ControlSummary?
        let samples = try NullCalibration.baseSamples(models.base, like: reference, prompt: prompt,
                                                      count: controls.isEmpty ? c.baseSampleControls : 0,
                                                      sampler: c.controlSampler, schedule: schedule)
        evaluations += samples.count * c.controlSampler.steps
        let all = controls + samples
        if !all.isEmpty {
            var masses: [Double] = [], snaps: [Double] = []
            for (i, control) in all.enumerated() {
                progress?("negative control \(i + 1)/\(all.count)")
                let (_, d, k) = try measure(control)
                masses.append(d.maxMass)
                snaps.append(Self.membershipSnap(d, k))
            }
            controlSummary = ControlSummary(source: controls.isEmpty ? "base-samples" : "images", count: all.count,
                                            maxClusterMass: masses, membershipSnap: snaps)
        }

        // 5–6. Regions with their confirmation, then the verdict.
        let regions = detection.clusters.enumerated().map { i, cluster -> MemorizedRegion in
            let p = controlSummary.map { $0.regionP(cluster.mass) }
            let significance: MemorizedRegion.Significance = p.map { $0 <= c.alpha ? .significant : .notSignificant } ?? .uncalibrated
            let byLevel = levels.indices.map { confirmation.snap(cells: cluster.cells, level: $0) }
            let bounds = cluster.cells.map { reference.cellRect(row: $0 / reference.grid.cols, col: $0 % reference.grid.cols) }
                .reduce(CGRect.null) { $0.union($1) }
            return MemorizedRegion(id: i + 1, cells: cluster.cells, areaFraction: Double(cluster.cells.count) / Double(reference.grid.count),
                                   bounds: bounds, mass: cluster.mass, meanCollapse: cluster.meanCollapse,
                                   peakCollapse: cluster.peakCollapse, peakLogSNR: levels[cluster.peakLevel],
                                   promptCollapse: cluster.meanPromptCollapse, pValue: p, significance: significance,
                                   snap: byLevel[cluster.peakLevel], snapByLevel: byLevel)
        }
        let membershipSnap = Self.membershipSnap(detection, confirmation)
        let (verdict, reasons) = VerdictRule.decide(regions: regions, scout: scout, issues: models.issues,
                                                    negativeArea: detection.negativeArea, controls: controlSummary,
                                                    membershipSnap: membershipSnap, configuration: c)

        let counted = Set(regions.filter(\.counts).flatMap(\.cells))
        var layers = [
            HeatmapLayer(name: MemorizationHeatmap.primaryLayer,
                         summary: "Sustained collapse ρ_base − ρ_target inside counted regions (0 = as free as the baseline, 1 = pinned)",
                         values: detection.sustained.enumerated().map { counted.contains($0) ? max($1, 0) : 0 },
                         scaleMax: 1, thresholded: true),
            HeatmapLayer(name: "collapse-raw", summary: "Sustained collapse at every released position, before thresholding",
                         values: detection.sustained, scaleMax: 1, thresholded: false),
        ]
        if let ps = detection.promptSustained {
            layers.append(HeatmapLayer(name: "prompt-collapse",
                                       summary: "Sustained collapse ρ_uncond − ρ_cond under the target: what the prompt pins",
                                       values: ps, scaleMax: 1, thresholded: false))
        }
        layers.append(HeatmapLayer(name: "score-difference",
                                   summary: "Forward-only surrogate (ε̂_target − ε̂_base)², mean over levels — conflates learning with memorization",
                                   values: GridMath.mean(f.scoreDifference), scaleMax: 1, thresholded: false))

        let size = imageSize ?? reference.frame.size
        let heatmap = MemorizationHeatmap(imageWidth: Int(size.width), imageHeight: Int(size.height), frame: reference.frame,
                                          grid: reference.grid, threshold: c.collapseThreshold, layers: layers, regions: regions)
        let everything = Array(0..<reference.grid.count)
        let summaries = levels.indices.map { l -> LevelSummary in
            let released = (0..<reference.grid.count).filter { detection.released[l][$0] }
            let mean = GridMath.mean(f.collapse[l])
            let prompted = f.promptCollapse.map { GridMath.mean($0[l]) }
            func avg(_ v: [Float]) -> Double { released.isEmpty ? 0 : released.reduce(0.0) { $0 + Double(v[$1]) } / Double(released.count) }
            return LevelSummary(logSNR: levels[l], releasedFraction: Double(released.count) / Double(reference.grid.count),
                                meanCollapse: avg(mean), maxCollapse: Double(released.map { mean[$0] }.max() ?? 0),
                                meanPromptCollapse: prompted.map(avg), snap: confirmation.snap(cells: everything, level: l))
        }
        return MemorizationResult(configuration: c, estimator: f.estimator, prompt: prompt, target: models.target.identifier,
                                  base: models.base.identifier, scout: scout, levels: summaries, heatmap: heatmap,
                                  membershipSnap: membershipSnap, frameSnap: confirmation.bandSnap(cells: everything),
                                  negativeArea: detection.negativeArea, controls: controlSummary, verdict: verdict,
                                  reasons: reasons, issues: models.issues, calibrated: false, schedule: schedule.info,
                                  evaluations: evaluations, seconds: Date().timeIntervalSince(started))
    }

    /// Confirmation band snap over every candidate region (the whole frame when there are none).
    static func membershipSnap(_ d: RegionDetection, _ k: ConfirmationField) -> Double {
        let cells = d.clusters.flatMap(\.cells)
        return k.bandSnap(cells: cells.isEmpty ? Array(0..<d.grid.count) : cells)
    }

    private func empty(models: ModelPair, reference: EncodedReference, prompt: ProbePrompt, scout: LevelScoutResult,
                       imageSize: CGSize?, estimator: String, schedule: SeedSchedule, evaluations: Int,
                       started: Date) -> MemorizationResult {
        let c = configuration
        let (verdict, reasons) = VerdictRule.decide(regions: [], scout: scout, issues: models.issues, negativeArea: 0,
                                                    controls: nil, membershipSnap: 1, configuration: c)
        let zeros = [Float](repeating: 0, count: reference.grid.count)
        let size = imageSize ?? reference.frame.size
        let heatmap = MemorizationHeatmap(
            imageWidth: Int(size.width), imageHeight: Int(size.height), frame: reference.frame, grid: reference.grid,
            threshold: c.collapseThreshold,
            layers: [HeatmapLayer(name: MemorizationHeatmap.primaryLayer, summary: "not measured", values: zeros, scaleMax: 1,
                                  thresholded: true)],
            regions: [])
        return MemorizationResult(configuration: c, estimator: estimator, prompt: prompt, target: models.target.identifier,
                                  base: models.base.identifier, scout: scout, levels: [], heatmap: heatmap, membershipSnap: 1,
                                  frameSnap: 1, negativeArea: 0, controls: nil, verdict: verdict, reasons: reasons,
                                  issues: models.issues, calibrated: false, schedule: schedule.info, evaluations: evaluations,
                                  seconds: Date().timeIntervalSince(started))
    }
}
