//
//  Verdict.swift
//  ScorpionKit
//
//  Step 6 — the readout. Rule-based and uncalibrated (like every Scorpion statistic until a
//  labeled set exists), with "the test was invalid" kept apart from "the test found nothing":
//
//  inconclusive             the test couldn't be run validly (no released levels, a backend
//                           issue such as an adapter trained on another base, too few
//                           controls for the profile, or a baseline that is freer than the
//                           target almost everywhere)
//  memorized                a counted region whose confirmation snap ≥ threshold
//  memorized-other-content  a counted region the target pins onto something else
//  no-evidence              nothing the controls don't also show — not evidence of non-use
//

import Foundation

public enum VerdictRule {
    public static func decide(regions: [MemorizedRegion], scout: LevelScoutResult, issues: [String],
                              negativeArea: Double, controls: ControlSummary?, membershipSnap: Double,
                              configuration c: MemorizationConfiguration) -> (MemorizationVerdict, [String]) {
        var reasons: [String] = []
        var invalid = issues
        if scout.selected.isEmpty {
            invalid.append("the base never releases the reference at the scouted noise levels — nothing to compare against")
        }
        if (controls?.count ?? 0) < c.minimumControls {
            invalid.append("profile '\(c.profile)' needs ≥ \(c.minimumControls) negative controls (got \(controls?.count ?? 0))")
        }
        if negativeArea > c.negativeAreaLimit {
            invalid.append(String(format: "the target is freer than the base over %.0f%% of released positions — the baseline is not underfitted here",
                                  100 * negativeArea))
        }
        if !invalid.isEmpty { return (.inconclusive, invalid) }

        let counted = regions.filter(\.counts)
        let dropped = regions.count - counted.count
        if dropped > 0 {
            reasons.append("\(dropped) candidate region(s) not significant against \(controls?.count ?? 0) controls")
        }
        guard let lead = counted.first else {
            reasons.append(String(format: "no region pinned (sustained collapse ≥ %.2f) beyond what the controls show", c.collapseThreshold))
            reasons.append("absence of memorization is not evidence the image was not used in training")
            return (.noEvidence, reasons)
        }
        let area = counted.reduce(0) { $0 + $1.areaFraction }
        reasons.append(String(format: "%d region(s) pinned, %.0f%% of the frame; strongest: mean collapse %.2f (peak %.2f) around λ = %.0f",
                              counted.count, 100 * area, lead.meanCollapse, lead.peakCollapse, lead.peakLogSNR))
        if let p = lead.pValue {
            reasons.append(String(format: "region mass %.1f, p = %.3f against %d controls", lead.mass, p, controls?.count ?? 0))
        } else {
            reasons.append("uncalibrated: no negative controls were run")
        }
        if let pc = lead.promptCollapse, abs(pc) >= Double(c.collapseThreshold) {
            reasons.append(String(format: "the prompt pins it too (prompt collapse %.2f vs unconditional)", pc))
        }
        if let controls {
            reasons.append(String(format: "membership snap %.1f (p = %.3f against controls)", membershipSnap, controls.snapP(membershipSnap)))
        }
        if lead.snap >= c.snapThreshold {
            reasons.append(String(format: "on independent draws the target returns the region to the reference %.1f× tighter than the base", lead.snap))
            return (.memorized, reasons)
        }
        if lead.snap >= 1 {
            reasons.append(String(format: "but it returns the region to the reference only %.1f× tighter than the base (< %.0f×)",
                                  lead.snap, c.snapThreshold))
        } else {
            reasons.append(String(format: "it pulls the region onto different content (%.1f× farther from the reference than the base)",
                                  1 / max(lead.snap, 1e-9)))
        }
        return (.memorizedOtherContent, reasons)
    }
}
