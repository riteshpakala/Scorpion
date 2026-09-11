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
    static func bits(_ v: Double) -> String { String(format: "%+.1f", v) }

    static func report(_ r: LikenessReport) -> String {
        var out: [String] = []
        let score = Int((r.harmScore * 100).rounded())
        out.append("")
        out.append("Scorpion \(r.scorpionVersion) · likeness harm score: \(score)/100\(r.calibrated ? "" : " (provisional, uncalibrated)")")
        out.append(String(repeating: "─", count: 72))
        out.append("Model      \(r.source.repo)\(r.source.commit.map { " @ \($0.prefix(10))" } ?? "")")
        let fraction = r.source.fetchedFraction.map { " (\(pct($0)) of \(ByteFormat.string(r.source.weightBytes)) weights)" } ?? ""
        out.append("Fetched    \(ByteFormat.string(r.source.bytesFetched)) over \(r.source.requests) requests\(fraction); \(ByteFormat.string(r.source.bytesFromCache)) from cache")
        out.append("Reference  \(r.reference.name) · \(r.reference.width)×\(r.reference.height) · id \(r.reference.referenceID.prefix(12)) · \(r.reference.faces.count) face(s)")
        if !r.flags.isEmpty { out.append("Flags      \(r.flags.joined(separator: ", "))") }

        out.append("")
        out.append("Components")
        for c in r.components {
            out.append(String(format: "  %-20@ %5.2f  (w %.2f)  %@", c.kind.rawValue as NSString, c.value, c.weight, c.note))
        }
        if r.components.isEmpty { out.append("  none available (no metadata, no analyzable weights)") }

        for u in r.units {
            out.append("")
            out.append("Unit       \(u.unit) · \(u.format.rawValue) · \(u.analyzedModules)/\(u.moduleCount) modules · family \(u.familyKey)")
            out.append("           conditioning heuristic: \(u.conditioning.method.rawValue)\(u.conditioning.dims.isEmpty ? "" : " dims \(u.conditioning.dims)") · \(u.conditioning.candidateCount) candidates")
            if let a = u.aggregate {
                out.append(String(format: "           narrowness %.2f · mean effective rank %.1f · mean concentration %.2f", a.narrowness, a.meanEffectiveRank, a.meanConcentration))
            }
            for m in u.topModules.prefix(3) {
                let name = m.module.count > 58 ? "…" + m.module.suffix(57) : m.module
                out.append(String(format: "           %@  ‖ΔW‖ %.3g  eff. rank %.1f/%d  σ₁²/Σσ² %.2f", name, m.frobeniusNorm, m.effectiveRank, m.spectrumSize, m.concentration))
            }
            for n in u.notes { out.append("           note: \(n)") }
        }

        let md = r.metadata
        out.append("")
        out.append("Metadata   sources: \(md.sources.isEmpty ? "none" : md.sources.joined(separator: ", "))")
        if !md.trainingTags.isEmpty {
            out.append("           training tags: " + md.trainingTags.prefix(10).map { "\($0.tag) (\($0.count))" }.joined(separator: ", "))
        }
        if !md.triggerWords.isEmpty { out.append("           triggers: \(md.triggerWords.joined(separator: ", "))") }
        if let b = md.baseModel { out.append("           base: \(b)") }
        if let m = r.metadataMatch, !m.matches.isEmpty {
            out.append("           matched: " + m.matches.prefix(6).map { String(format: "%@↔%@ %.2f", $0.descriptor, $0.term, $0.similarity) }.joined(separator: ", "))
        }
        if !r.reference.descriptors.isEmpty {
            out.append("")
            out.append("Described  " + r.reference.descriptors.filter { $0.category != "tags" }.prefix(10)
                .map { String(format: "%@ %.2f", $0.phrase, $0.probability) }.joined(separator: " · "))
        }
        if !r.bankNeighbors.isEmpty {
            out.append("Bank       " + r.bankNeighbors.prefix(3).map { String(format: "%@ %.2f (%@)", $0.label, $0.similarity, $0.source) }.joined(separator: "; "))
        }
        out.append("")
        out.append("Probes     \(r.probes.note)")
        out.append("Seeds      schedule \(r.reference.seedSchedule.id) (\(r.reference.seedSchedule.version)) · \(r.reference.permutations) permutations · \(r.reference.band)")
        out.append("Prompts    \"\(r.reference.prompts.attacker)\"")
        out.append("")
        out.append(r.disclaimer)
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

    static func reference(_ a: ReferenceAnalysis, show: Int) -> String {
        var out: [String] = []
        out.append("")
        out.append("\(a.name) · \(a.width)×\(a.height) · id \(a.referenceID)")
        for f in a.faces {
            out.append(String(format: "  face%d  %.0f,%.0f %.0f×%.0f  conf %.2f%@", f.index, f.bounds.minX, f.bounds.minY, f.bounds.width, f.bounds.height,
                              f.confidence, f.index == a.selectedFace ? "  ← selected" : ""))
        }
        for seg in a.segmentation {
            out.append(String(format: "  face%d segmented by %@ · mask covers %.1f%% of the image · features: %@", seg.faceIndex, seg.method,
                              (seg.areaFraction ?? 0) * 100, seg.features.joined(separator: ", ")))
        }
        let s = a.visualStats
        out.append(String(format: "  luminance entropy %.2f bits · spectral slope %.2f · pHash %@", s.luminanceEntropyBits, s.spectralSlope, s.perceptualHash))
        if !a.descriptors.isEmpty {
            out.append("")
            out.append("Descriptors (\(a.clipModel ?? "CLIP"))")
            for d in a.descriptors {
                out.append(String(format: "  %-13@ %-32@ %.2f  [%@]", d.category as NSString, d.phrase as NSString, d.probability, d.source))
            }
        }
        out.append("")
        out.append("Seed plan: schedule \(a.seedPlan.schedule.id) · \(a.seedPlan.permutations.count) permutations · \(a.seedPlan.band) · crops: \(a.crops.map(\.label).joined(separator: ", "))")
        for p in a.seedPlan.permutations.prefix(show) {
            out.append(String(format: "  #%-3d λ=%+6.2f  ε key %016llx  crop %d  \"%@\"", p.index, p.logSNR, p.noiseKey, p.cropIndex, p.prompt))
            out.append("        null: \"\(p.nullPrompt)\"")
        }
        for n in a.notes { out.append("note: \(n)") }
        return out.joined(separator: "\n")
    }

    static func mu(_ e: SubsetSimulation.Result) -> String {
        let v = String(format: "%.2e", e.probability)
        return e.reachedTarget ? v : "< \(v)\(e.stalled ? " (stalled)" : "")"
    }

    /// 90% interval on μ, from the subset-simulation CoV.
    static func muCI(_ e: SubsetSimulation.Result) -> String {
        guard e.reachedTarget else { return "" }
        let ci = e.log2CI90
        return String(format: "[%.1e, %.1e]", pow(2, ci[0]), pow(2, ci[1]))
    }

    static func needle(_ r: NeedleResearchReport) -> String {
        var out: [String] = []
        let o = r.options
        out.append("")
        out.append("Scorpion research · face-seed needle (analytic models; true needle mass known)")
        let seg = r.segmentation.map { "\($0.method)\($0.usedPersonMatte ? ", person matte" : "")" } ?? "no face — center region"
        out.append("reference \(r.referenceID.prefix(12)) · segmentation: \(seg) · \(r.referencePhotos) photo(s) + \(r.syntheticPhotos) synthetic · \(r.controls) control(s)")
        out.append(String(format: "seed segment: %@ covers %.1f%% of the seed (face %.1f%%) · captures %.0f%% of influence · one-step vs through-sampler IoU %.2f",
                          r.footprint.method, r.footprint.fraction * 100, r.faceFraction * 100, r.footprint.capturedInfluence * 100,
                          r.footprintIoUOneStepVsSampler))
        out.append("")
        out.append(String(format: "True needle mass π = %.1e  (%.1f bits of luck under the target) · sampler %@ %d steps",
                          r.truth, -log2(r.truth), o.needle.sampler.solver.rawValue, o.needle.sampler.steps))
        out.append("  criterion              w     μ target          μ base            Δbits          ι      reach T/B   μ target 90% CI")
        for (label, res) in [("identity oracle", r.oracle), ("latent fidelity", r.fidelity)] {
            for (t, b) in zip(res.target, res.base) {
                let d = res.deltaBits.first { $0.guidance == t.guidance }
                let delta = d.map { ($0.isLowerBound ? "≥ " : "") + String(format: "%+.1f", $0.value) } ?? "—"
                let iota = t.backgroundIndependence.map { String(format: "%.2f", $0) } ?? "—"
                let reach = res.reachability.map { String(format: "%.2f/%.2f", $0.targetHitRate, $0.baseHitRate) } ?? "—"
                out.append("  \(label.padding(toLength: 20, withPad: " ", startingAt: 0)) \(String(format: "%-5.1f", t.guidance)) \(mu(t.estimate).padding(toLength: 17, withPad: " ", startingAt: 0)) \(mu(b.estimate).padding(toLength: 17, withPad: " ", startingAt: 0)) \(delta.padding(toLength: 14, withPad: " ", startingAt: 0)) \(iota.padding(toLength: 6, withPad: " ", startingAt: 0)) \(reach.padding(toLength: 11, withPad: " ", startingAt: 0)) \(muCI(t.estimate))")
            }
        }
        if let bf = r.bruteForceFidelity {
            out.append(String(format: "  brute-force Monte Carlo, fidelity event under the target: %.2e", bf))
        }
        out.append(String(format: "  latent fidelity vs oracle on %d target samples: accepts %.0f%% of identity samples, %.2f%% of others",
                          r.agreement.samples, r.agreement.tpr * 100, r.agreement.fpr * 100))
        if let t = r.oracle.target.first, !t.featureRetention.isEmpty {
            out.append("  feature retention (hits surviving a resampled feature segment): "
                       + t.featureRetention.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", "))
        }
        out.append("")
        out.append("Attack curve (P[success within N seeds], oracle): target / base")
        for p in r.oracle.attackCurve {
            out.append(String(format: "  w=%.1f  N=%-6d  %.3g / %@%.3g", p.guidance, p.attempts, p.target, p.baseIsBound ? "≤ " : "", p.base))
        }
        out.append("reach T/B: the inverted reference seed's face segment with fresh surroundings reproduces the face (numerical; a")
        out.append("           needle's basin can be narrower than inversion error — every image is reachable only in exact arithmetic)")
        if let e = r.oracle.earlyExitAgreement { out.append(String(format: "Early exit at λ=%.1f agrees with full sampling on %.1f%% of decisions", o.needle.earlyExitLogSNR ?? 0, e * 100)) }
        return out.joined(separator: "\n")
    }

    static func memory(_ r: MemoryResearchReport) -> String {
        var out: [String] = []
        let o = r.options
        out.append("")
        out.append("Scorpion research · memory pools (analytic models; memorization strength κ known)")
        let seg = r.segmentation.map { "\($0.method)\($0.usedPersonMatte ? ", person matte" : "")" } ?? "no face — center region"
        let gate = o.trigger.map { String(format: "trigger \"%@\" (unconditional keeps %.0e)", $0, o.leak) } ?? "ungated"
        out.append("reference \(r.referenceID.prefix(12)) · segmentation: \(seg) · \(r.referencePhotos) photo(s) · \(o.members) training photos"
                   + "\(o.duplicates > 1 ? " (first ×\(o.duplicates))" : "") · scope \(o.scope.rawValue) · \(gate)")
        out.append("κ = 0: the identity is learned (one smooth component); κ = 1: the training photos are stored (pools of width h²)")
        out.append("")
        out.append("       │ your photo was a training member                           │ your photo was held out")
        out.append("κ      │ tier               collapse   λ    snap    IoU   β½ T/B    │ tier               collapse  recur")
        for row in r.rows {
            let m = row.member, h = row.heldOut
            let mb = m.basin.map { String(format: "%.2f/%.2f", $0.halfRadiusTarget, $0.halfRadiusBase) } ?? "—"
            let recur = h.capture.first?.nearDuplicateRate.map { String(format: "%.2f", $0) } ?? "—"
            out.append(String(format: "%.2f   │ %@ %6.2f  %4.0f  %6.1f  %5.2f   %@ │ %@ %6.2f    %@",
                              row.kappa, m.tier.rawValue.padding(toLength: 18, withPad: " ", startingAt: 0) as NSString,
                              m.peakSignal, m.peakLogSNR, m.peakSnap, m.iou,
                              mb.padding(toLength: 9, withPad: " ", startingAt: 0) as NSString,
                              h.tier.rawValue.padding(toLength: 18, withPad: " ", startingAt: 0) as NSString, h.peakSignal,
                              recur as NSString))
        }
        out.append("")
        out.append("Audit (other photos, Hutchinson-4, λ 2…8) and truth (member world)")
        out.append("κ      membership AUC   in a pool: member/held-out/look-alike/stranger   exact collapse (err)   map IoU   face Δbits (exact)")
        for row in r.rows {
            let err = row.collapseError.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
            let rates = ["member", "held-out", "look-alike", "stranger"]
                .map { row.poolRate[$0].map { String(format: "%.0f%%", $0 * 100) } ?? "—" }.joined(separator: " / ")
            out.append(String(format: "%.2f   %@             %@   %5.2f (%@)          %5.2f     %+.1f (%+.1f)", row.kappa,
                              (row.membershipAUC.map { String(format: "%.2f", $0) } ?? "—  ") as NSString,
                              rates.padding(toLength: 34, withPad: " ", startingAt: 0) as NSString,
                              row.exactPeakCollapse, err.padding(toLength: 4, withPad: " ", startingAt: 0) as NSString,
                              row.exactIoU, row.faceDeltaBits, row.exactBits))
        }
        if let strongest = r.rows.max(by: { $0.kappa < $1.kappa }) {
            out.append("")
            out.append(String(format: "Detail at κ = %.2f, your photo as a member (%@)", strongest.kappa, strongest.member.method))
            out.append(memoryDetail(strongest.member))
            out.append("")
            out.append(String(format: "Detail at κ = %.2f, your photo held out", strongest.kappa))
            out.append(memoryDetail(strongest.heldOut))
        }
        out.append("")
        out.append("collapse: ρ_base − ρ_target on the face (retained variance; 0 = free, 1 = pinned) · snap: how much tighter the target")
        out.append("returns your noised photo to itself · β½: seed-space radius around your photo's seed endpoint that still returns it")
        out.append("recur: share of the person's generations that come back as near-copies from different seeds (stored photos)")
        return out.joined(separator: "\n")
    }

    static func memoryDetail(_ m: MemoryPoolResult) -> String {
        var out: [String] = []
        out.append("  tier \(m.tier.rawValue)" + (m.collapseBand.map { String(format: " · collapse band λ %.0f…%.0f", $0[0], $0[1]) } ?? ""))
        for reason in m.reasons { out.append("   · \(reason)") }
        out.append("  λ      ρ face T/B      collapse  bg      prompt   snap      person T/B")
        for p in m.points {
            let rho = p.retainedTarget.map { t in String(format: "%.2f/%.2f", t.face, p.retainedBase?.face ?? .nan) }
                ?? String(format: "%.2f/%.2f", p.spreadTarget.face, p.spreadBase.face)
            let prompt = (p.promptCollapse?.face).map { String(format: "%+.2f", $0) } ?? "  —  "
            let person = p.identityTarget.map { String(format: "%.2f/%.2f", $0, p.identityBase ?? 0) } ?? "—"
            out.append(String(format: "  %+4.0f   %@   %+.2f     %+.2f   %@    %7.1f   %@", p.logSNR,
                              rho.padding(toLength: 13, withPad: " ", startingAt: 0) as NSString, p.signal, p.backgroundSignal,
                              prompt as NSString, p.snap, person as NSString))
        }
        if let b = m.basin {
            out.append("  basin r(β), \(b.sampler): " + zip(b.betas, zip(b.target, b.base))
                .map { String(format: "%.2f→%.2f/%.2f", $0.0, $0.1.0, $0.1.1) }.joined(separator: "  "))
        }
        for c in m.capture {
            out.append(String(format: "  capture %@ prompt: μ target %.2e%@ · base %@%.2e · Δ %@%+.1f bits · recurring copies %@",
                              c.kind, c.target, c.targetReached ? "" : " (bound)", c.baseIsBound ? "≤ " : "", c.base,
                              c.baseIsBound ? "≥ " : "", c.deltaBits,
                              (c.nearDuplicateRate.map { String(format: "%.2f", $0) } ?? "—") as NSString))
        }
        if !m.featureShare.isEmpty {
            out.append("  feature share of the face's collapse: "
                       + m.featureShare.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", "))
        }
        return out.joined(separator: "\n")
    }

    static func experiment(_ r: ExperimentReport) -> String {
        var out: [String] = []
        out.append("")
        out.append("Scorpion experiment · \(r.name) · \(r.rows.isEmpty ? "needle arm not run" : (r.go ? "GO" : "NO-GO"))")
        if !r.rows.isEmpty {
            out.append("  pair                                              label           needle Δbits   face Δbits   ι      ±bits   calls")
        }
        for row in r.rows {
            let nd = (row.needleIsLowerBound ? "≥" : " ") + String(format: "%+.1f", row.needleDeltaBits)
            out.append("  \(String(row.pair.prefix(48)).padding(toLength: 49, withPad: " ", startingAt: 0)) \(row.label.rawValue.padding(toLength: 15, withPad: " ", startingAt: 0)) \(nd.padding(toLength: 14, withPad: " ", startingAt: 0)) \(String(format: "%+10.1f", row.faceDeltaBits))   \(row.backgroundIndependence.map { String(format: "%.2f", $0) } ?? "—  ")   \(String(format: "%.2f", row.ciHalfWidthBits))   \(row.denoiserCalls)")
        }
        if !r.rows.isEmpty {
            out.append("")
            for (k, v) in r.auc.sorted(by: { $0.key < $1.key }) { out.append(String(format: "AUC %@: %.3f", k, v)) }
            out.append("")
        }
        for c in r.criteria {
            out.append("\(c.pass ? "PASS" : "FAIL")  \(c.name): \(c.value)   (\(c.requirement))")
        }
        if let rows = r.memoryRows, let criteria = r.memoryCriteria {
            out.append("")
            out.append("Tier M · memory pools · \((r.memoryGo ?? false) ? "GO" : "NO-GO")")
            out.append("  κ      label      n   median collapse   median band snap   in a pool   median IoU")
            let kappas = Array(Set(rows.map(\.kappa))).sorted()
            let pools: Set<MemoryPoolResult.Tier> = [.memorizedPhoto, .identityPool, .nearbyPool]
            for k in kappas {
                for label in [ExperimentManifest.Label.member, .heldOut] {
                    let g = rows.filter { $0.kappa == k && $0.label == label }
                    guard !g.isEmpty else { continue }
                    func med(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
                    out.append(String(format: "  %.2f   %@ %3d   %8.2f          %10.1f          %4.0f%%       %5.2f", k,
                                      label.rawValue.padding(toLength: 9, withPad: " ", startingAt: 0) as NSString, g.count,
                                      med(g.map(\.collapse)), med(g.map(\.snap)),
                                      100 * Double(g.filter { pools.contains($0.tier) }.count) / Double(g.count), med(g.map(\.iou))))
                }
            }
            for c in criteria { out.append("\(c.pass ? "PASS" : "FAIL")  \(c.name): \(c.value)   (\(c.requirement))") }
        }
        for n in r.notes { out.append("note: \(n)") }
        return out.joined(separator: "\n")
    }

    static func research(_ r: GMMResearchReport) -> String {
        var out: [String] = []
        let o = r.options
        out.append("")
        out.append("Scorpion research · analytic Gaussian-mixture backend (closed-form ground truth)")
        out.append("reference \(r.referenceID.prefix(12)) · latent \(o.side)×\(o.side) · \(o.components) base components · \(o.permutations) seed permutations · \(o.band)")
        out.append("region \(r.regionLabel): \(pct(r.regionAreaFraction)) of the latent · seed schedule \(r.seedSchedule.id)")
        out.append("")
        out.append("Likelihood ratio vs. base (bits of the reference in the weights beyond the base)")
        out.append("  variant          exact Δbits   estimated Δbits (90% CI)          region share   z*-atypicality (region)")
        for v in r.variants {
            let ci = String(format: "[%+.1f, %+.1f]", v.estimate.ci90[0], v.estimate.ci90[1])
            let share = v.region?.share.map { pct($0) } ?? "—"
            let typ = v.inversion.first { $0.region != "full" }.map { String(format: "%.1f", $0.atypicality) } ?? "—"
            out.append("  \(v.name.padding(toLength: 16, withPad: " ", startingAt: 0)) \(bits(v.exactDeltaBits).padding(toLength: 12, withPad: " ", startingAt: 0))  \(bits(v.estimate.deltaBits).padding(toLength: 8, withPad: " ", startingAt: 0)) \(ci.padding(toLength: 24, withPad: " ", startingAt: 0))  \(share.padding(toLength: 13, withPad: " ", startingAt: 0))  \(typ)")
        }
        if let base = r.baseInversion.first(where: { $0.region != "full" }) {
            out.append(String(format: "  base model z*-atypicality (region): %.1f   (0 = typical Gaussian noise)", base.atypicality))
        }
        if let learned = r.variants.first, !learned.estimate.subBands.isEmpty {
            out.append("")
            out.append("Where the bits live (with-reference, full latent, by log-SNR sub-band)")
            for s in learned.estimate.subBands {
                out.append(String(format: "  λ∈[%+5.1f, %+5.1f)  %+8.1f bits  (%d samples)", s.lo, s.hi, s.deltaBits, s.samples))
            }
        }
        out.append("")
        if let c = r.crnVarianceRatio {
            out.append(String(format: "Common random numbers: paired estimator variance is %.1f%% of unpaired (%.1f× fewer seeds for the same precision)", c * 100, 1 / max(c, 1e-9)))
        }
        func sens(_ s: RegionSensitivityResult) -> String {
            String(format: "%.1f%% of ∂x̂₀/∂ε mass inside a region covering %.1f%%%@", s.insideMassFraction * 100, s.areaFraction * 100,
                   s.concentration.map { String(format: " (concentration %.1f×)", $0) } ?? "")
        }
        out.append("Seed contribution to the region, separable model: \(sens(r.sensitivitySeparable))")
        out.append("Seed contribution to the region, coupled model:   \(sens(r.sensitivityCoupled))")
        return out.joined(separator: "\n")
    }
}
