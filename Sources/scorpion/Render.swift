//
//  Render.swift
//  scorpion
//
//  Human-readable terminal output.
//

import Foundation
import ScorpionKit

enum Render {
    static func pct(_ v: Double) -> String { String(format: "%.1f%%", v * 100) }
    static func rule() -> String { String(repeating: "─", count: 78) }
    /// Left-aligned to `n` columns (String(format:) doesn't pad %@).
    static func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
    /// Right-aligned to `n` columns.
    static func lpad(_ s: String, _ n: Int) -> String { s.count >= n ? s : String(repeating: " ", count: n - s.count) + s }

    static func verdictTitle(_ v: MemorizationVerdict) -> String {
        switch v {
        case .memorized: return "MEMORIZED"
        case .memorizedOtherContent: return "Memorized — other content"
        case .noEvidence: return "No evidence of memorization"
        case .inconclusive: return "Inconclusive"
        }
    }

    static func memorization(_ report: MemorizationReport) -> String {
        let r = report.result
        var out: [String] = [""]
        out.append("Scorpion \(report.scorpionVersion) · \(verdictTitle(r.verdict)) (rule-based, uncalibrated)")
        out.append(rule())
        out.append("Reference  \(report.reference.name) · \(report.reference.width)×\(report.reference.height) · sha256 \(report.reference.id.prefix(16))…")
        out.append("Model      \(report.backend)\(report.model.map { " · \($0)" } ?? "") · target \(r.target) vs base \(r.base)")
        for (k, v) in report.provenance.sorted(by: { $0.key < $1.key }) { out.append("           \(k): \(v)") }
        out.append("Prompt     \(r.prompt.isConditional ? "\"\(r.prompt.text)\" [\(r.prompt.source)]" : "(unconditional)")")
        out.append(String(format: "Probe      profile %@ · %@ · %d draws · grid %@ · %d evaluations · %.1f s",
                          r.configuration.profile, r.estimator, r.configuration.draws, r.heatmap.grid.description,
                          r.evaluations, r.seconds))
        out.append("Levels     " + r.scout.levels.map { l in
            String(format: "λ%@%g %.0f%%", l.selected ? "*" : " ", l.logSNR, 100 * l.releasedFraction)
        }.joined(separator: "  ") + "   (* measured; % = positions the base releases)")
        if !r.levels.isEmpty {
            out.append("")
            out.append("  λ      released   mean collapse   max collapse   prompt collapse   frame snap")
            for l in r.levels {
                out.append(String(format: "  %@  %7.0f%%   %13.3f   %12.3f   %@   %10.2f", pad(String(format: "%g", l.logSNR), 5),
                                  100 * l.releasedFraction, l.meanCollapse, l.maxCollapse,
                                  lpad(l.meanPromptCollapse.map { String(format: "%.3f", $0) } ?? "—", 15), l.snap))
            }
        }
        out.append("")
        if r.regions.isEmpty {
            out.append("Regions    none above τ = \(r.configuration.collapseThreshold)")
        } else {
            out.append("  #   area    mean   peak   λ    mass     p        significance     snap    bounds (px)")
            for g in r.regions {
                let b = g.bounds
                out.append(String(format: "  %-2d  %5.1f%%  %5.2f  %5.2f  %@  %7.1f  %@  %@  %6.1f  %.0f,%.0f %.0f×%.0f",
                                  g.id, 100 * g.areaFraction, g.meanCollapse, g.peakCollapse, pad(String(format: "%g", g.peakLogSNR), 3),
                                  g.mass, pad(g.pValue.map { String(format: "%.3f", $0) } ?? "—", 7), pad(g.significance.rawValue, 15),
                                  g.snap, b.minX, b.minY, b.width, b.height))
            }
        }
        if let c = r.controls {
            out.append(String(format: "Controls   %d (%@) · max cluster mass median %.1f · membership snap %.2f (p %.3f)", c.count, c.source,
                              Statistics.median(c.maxClusterMass) ?? 0, r.membershipSnap, c.snapP(r.membershipSnap)))
        } else {
            out.append(String(format: "Controls   none — regions are uncalibrated · membership snap %.2f", r.membershipSnap))
        }
        if let b = report.basin {
            out.append(String(format: "Seed basin region %d · β½ target %.2f vs base %.2f · r(0) %.2f/%.2f · r(1) %.2f/%.2f · %@",
                              b.region, b.halfRadiusTarget, b.halfRadiusBase, b.target.first ?? 0, b.base.first ?? 0,
                              b.target.last ?? 0, b.base.last ?? 0, b.sampler))
        }
        out.append("")
        out.append("Verdict    \(verdictTitle(r.verdict))")
        for reason in r.reasons { out.append("  · \(reason)") }
        out.append("")
        out.append(report.disclaimer)
        return out.joined(separator: "\n")
    }

    static func inspection(_ i: ModelInspection) -> String {
        var out: [String] = []
        out.append("")
        out.append("\(i.source.repo)\(i.source.commit.map { " @ \($0.prefix(10))" } ?? "") · \(ByteFormat.string(i.source.weightBytes)) of weights · headers fetched: \(ByteFormat.string(i.source.bytesFetched)) in \(i.source.requests) requests")
        for f in i.source.files {
            out.append("  \(f.format.rawValue.padding(toLength: 11, withPad: " ", startingAt: 0)) \(f.path)  \(f.size.map(ByteFormat.string) ?? "?")\(f.tensorCount.map { "  \($0) tensors" } ?? "")\(f.note.map { "  — \($0)" } ?? "")")
        }
        for u in i.units {
            out.append("")
            out.append("Unit \(u.unit): \(u.format.rawValue), \(u.moduleCount) modules, family \(u.familyKey)")
            out.append("  conditioning: \(u.conditioning.method.rawValue)\(u.conditioning.dims.isEmpty ? "" : " dims \(u.conditioning.dims)"), \(u.conditioning.candidateCount) candidates")
            out.append("  plan: \(u.plannedModules) modules, \(ByteFormat.string(u.plannedBytes)) of \(ByteFormat.string(u.unitBytes))")
            for m in u.sampleModules { out.append("    \(m)") }
            if !u.trainingMetadataKeys.isEmpty {
                out.append("  header metadata: \(u.trainingMetadataKeys.prefix(12).joined(separator: ", "))\(u.trainingMetadataKeys.count > 12 ? ", …" : "")")
            }
        }
        for (path, s) in i.ggufSummaries.sorted(by: { $0.key < $1.key }) {
            out.append("GGUF \(path): " + s.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))
        }
        let md = i.metadata
        out.append("")
        out.append("Metadata sources: \(md.sources.isEmpty ? "none" : md.sources.joined(separator: ", "))")
        if !md.trainingTags.isEmpty { out.append("  training tags: " + md.trainingTags.prefix(12).map { "\($0.tag) (\($0.count))" }.joined(separator: ", ")) }
        if !md.triggerWords.isEmpty { out.append("  triggers: \(md.triggerWords.joined(separator: ", "))") }
        if let b = md.baseModel { out.append("  base: \(b)") }
        if !md.personIndicators.isEmpty { out.append("  person indicators: \(md.personIndicators.joined(separator: ", "))") }
        if !i.flags.isEmpty { out.append("Flags: \(i.flags.joined(separator: ", "))") }
        return out.joined(separator: "\n")
    }

    static func memoryResearch(_ r: MemoryResearchReport) -> String {
        var out: [String] = [""]
        out.append("Memorization probe on analytic worlds · grid \(r.side)×\(r.side) · planted region \(r.plantedCells) cells · reference \(r.referenceID.prefix(12))…")
        out.append(rule())
        out.append("κ      │ reference was a training member                         │ reference held out")
        out.append("       │ verdict                 mean  λ     snap   IoU  exIoU │ verdict                 mean")
        for row in r.rows {
            let m = row.member, h = row.heldOut
            func mean(_ x: MemorizationResult) -> String { x.regions.first.map { String(format: "%.2f", $0.meanCollapse) } ?? "0.00" }
            out.append(String(format: "%.2f   │ %@  %@  %@  %5.1f  %4.2f  %4.2f │ %@  %@",
                              row.kappa, pad(m.verdict.rawValue, 22), mean(m),
                              pad(m.regions.first.map { String(format: "%g", $0.peakLogSNR) } ?? "—", 3),
                              m.regions.first?.snap ?? m.membershipSnap, row.iou, row.exactIoU, pad(h.verdict.rawValue, 22), mean(h)))
        }
        out.append("")
        out.append("κ      membership AUC   flagged: member  held-out  look-alike  stranger   exact collapse  estimated")
        for row in r.rows {
            func rate(_ k: String) -> String { row.flagRate[k].map { pct($0) } ?? "—" }
            out.append(String(format: "%.2f   %@   %@  %@  %@  %@   %14.2f  %@", row.kappa,
                              lpad(row.membershipAUC.map { String(format: "%.2f", $0) } ?? "—", 14), lpad(rate("member"), 15),
                              lpad(rate("held-out"), 8), lpad(rate("look-alike"), 10), lpad(rate("stranger"), 8),
                              row.exactPeakCollapse, lpad(row.estimatedPeakCollapse.map { String(format: "%.2f", $0) } ?? "—", 9)))
        }
        return out.joined(separator: "\n")
    }

    static func experiment(_ r: ExperimentReport) -> String {
        var out: [String] = [""]
        out.append("Scorpion experiment · \(r.name) · \(r.go ? "GO" : "NO-GO")")
        out.append(rule())
        out.append("  κ      label      n   median collapse   median snap   flagged   median IoU")
        let groups = Dictionary(grouping: r.rows) { "\($0.kappa)|\($0.label.rawValue)" }
        for key in groups.keys.sorted() {
            let rows = groups[key]!
            let flagged = rows.filter { $0.verdict == .memorized || $0.verdict == .memorizedOtherContent }.count
            out.append(String(format: "  %.2f   %@  %2d   %15.2f   %11.1f   %6.0f%%   %10.2f", rows[0].kappa, pad(rows[0].label.rawValue, 9),
                              rows.count, Statistics.median(rows.map(\.collapse)) ?? 0, Statistics.median(rows.map(\.snap)) ?? 0,
                              100 * Double(flagged) / Double(rows.count), Statistics.median(rows.map(\.iou)) ?? 0))
        }
        for c in r.criteria { out.append("\(c.pass ? "PASS" : "FAIL")  \(c.name): \(c.value)   (\(c.requirement))") }
        for n in r.notes { out.append("note: \(n)") }
        return out.joined(separator: "\n")
    }
}
