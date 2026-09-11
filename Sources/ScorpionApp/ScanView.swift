//
//  ScanView.swift
//  ScorpionApp
//

import ScorpionKit
import SwiftUI

struct ScanView: View {
    @ObservedObject var model: ScanViewModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(alignment: .top, spacing: 0) {
                referencePane
                    .frame(minWidth: 300, idealWidth: 360, maxWidth: 420)
                Divider()
                resultsPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Divider()
            statusBar
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "scope").font(.title2).foregroundStyle(.secondary)
            TextField("Model link — huggingface.co/… or github.com/…", text: $model.modelLink)
                .textFieldStyle(.roundedBorder)
                .onSubmit { model.scan() }
            Button {
                model.scan()
            } label: {
                Label("Scan", systemImage: "play.fill").frame(minWidth: 70)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.canScan)
        }
        .padding(12)
    }

    // MARK: Reference

    private var referencePane: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reference").font(.headline)
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                    .foregroundStyle(.tertiary)
                if let image = model.image {
                    FaceOverlayImage(image: image, imageSize: model.imageSize, faces: model.faces,
                                     overlays: model.segmentOverlays, selected: $model.selectedFace)
                        .padding(8)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "photo.badge.plus").font(.system(size: 34)).foregroundStyle(.secondary)
                        Text("Drop an image or choose one").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(minHeight: 260)
            .dropDestination(for: URL.self) { urls, _ in
                guard let url = urls.first else { return false }
                model.loadImage(url)
                return true
            }

            HStack {
                Button("Choose Image…") { model.chooseImage() }
                Spacer()
                if let name = model.imageName {
                    Text(name).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            if model.faces.count > 1 {
                Text("\(model.faces.count) faces — click one to analyze it").font(.caption).foregroundStyle(.secondary)
            }
            if let method = model.segmentMethods[model.selectedFace] {
                Text("Face segmentation: \(method)").font(.caption).foregroundStyle(.secondary)
            }

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Fetch budget")
                        Spacer()
                        Text("\(Int(model.budgetMB)) MB").monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $model.budgetMB, in: 16...2048, step: 16)
                    Toggle("Describe with CLIP", isOn: $model.useCLIP)
                }
            }

            Button {
                model.runResearch()
            } label: {
                Label("Run theory harness", systemImage: "function")
            }
            .disabled(model.image == nil || model.isRunning)
            .help("Seed/band/region probes on analytic models built around this image, next to exact answers.")
            Spacer()
        }
        .padding(12)
    }

    // MARK: Results

    private var resultsPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let error = model.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
                if let report = model.report {
                    ReportView(report: report, export: model.exportReport)
                } else if model.isRunning {
                    VStack(spacing: 10) {
                        ProgressView()
                        Text(model.phase).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                } else if model.research == nil {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Paste a model link, add a reference image, and scan.").font(.title3)
                        Text("Scorpion reads tensor headers with byte-range requests, fetches only the matrices that read the conditioning signal, and scores how likely the model is to reproduce the reference's likeness — without downloading the whole model or generating images.")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: 520, alignment: .leading)
                    .padding(.top, 40)
                }
                if let research = model.research {
                    ResearchView(report: research)
                }
                if let needle = model.needle {
                    NeedleView(report: needle)
                }
                if let memory = model.memory {
                    MemoryView(report: memory)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Status

    private var statusBar: some View {
        HStack {
            Text(model.phase.isEmpty ? "Ready" : model.phase).lineLimit(1).truncationMode(.middle)
            Spacer()
            if model.bytesFetched > 0 || model.weightBytes != nil {
                FetchMeter(fetched: model.bytesFetched, total: model.weightBytes)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

// MARK: - Components

struct FetchMeter: View {
    let fetched: Int
    let total: Int?

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "arrow.down.circle")
            if let total, total > 0 {
                Text("\(ByteFormat.string(fetched)) of \(ByteFormat.string(total)) weights")
                ProgressView(value: min(Double(fetched) / Double(total), 1)).frame(width: 90)
                Text(String(format: "%.1f%%", 100 * Double(fetched) / Double(total))).monospacedDigit()
            } else {
                Text("\(ByteFormat.string(fetched)) fetched")
            }
        }
    }
}

struct FaceOverlayImage: View {
    let image: NSImage
    let imageSize: CGSize
    let faces: [FaceRegion]
    var overlays: [Int: NSImage] = [:]
    @Binding var selected: Int

    var body: some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / max(imageSize.width, 1), geo.size.height / max(imageSize.height, 1))
            let w = imageSize.width * scale, h = imageSize.height * scale
            let ox = (geo.size.width - w) / 2, oy = (geo.size.height - h) / 2
            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .resizable()
                    .frame(width: w, height: h)
                    .offset(x: ox, y: oy)
                if let overlay = overlays[selected] {
                    Image(nsImage: overlay)
                        .resizable()
                        .frame(width: w, height: h)
                        .offset(x: ox, y: oy)
                        .allowsHitTesting(false)
                }
                ForEach(faces, id: \.index) { face in
                    let r = face.bounds
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(face.index == selected ? Color.accentColor : Color.white.opacity(0.7),
                                lineWidth: face.index == selected ? 3 : 1.5)
                        .frame(width: r.width * scale, height: r.height * scale)
                        .offset(x: ox + r.minX * scale, y: oy + r.minY * scale)
                        .contentShape(Rectangle())
                        .onTapGesture { selected = face.index }
                }
            }
        }
    }
}

struct ReportView: View {
    let report: LikenessReport
    let export: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 18) {
                ScoreRing(value: report.harmScore)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Likeness harm score").font(.headline)
                    Text(report.calibrated ? "Calibrated" : "Provisional — uncalibrated").font(.subheadline).foregroundStyle(.secondary)
                    Text("\(report.source.repo)\(report.source.commit.map { " @ \($0.prefix(10))" } ?? "")")
                        .font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    FetchMeter(fetched: report.source.bytesFetched, total: report.source.weightBytes)
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Export JSON…", action: export)
            }
            if !report.flags.isEmpty {
                Label(report.flags.joined(separator: ", "), systemImage: "flag.fill").foregroundStyle(.orange)
            }

            ResultSection(title: "Components") {
                if report.components.isEmpty {
                    Text("No signals available (no metadata, no analyzable weights).").foregroundStyle(.secondary)
                }
                ForEach(Array(report.components.enumerated()), id: \.offset) { _, c in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text(c.kind.rawValue).font(.callout.weight(.medium))
                            Spacer()
                            Text(String(format: "%.2f", c.value)).monospacedDigit()
                        }
                        ProgressView(value: c.value)
                        Text(c.note).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
            }

            ForEach(Array(report.units.enumerated()), id: \.offset) { _, u in
                ResultSection(title: "Weights · \(u.unit)") {
                    Text("\(u.format.rawValue) · \(u.analyzedModules) of \(u.moduleCount) modules analyzed · family \(u.familyKey)")
                    Text("Conditioning heuristic: \(u.conditioning.method.rawValue)\(u.conditioning.dims.isEmpty ? "" : " (width \(u.conditioning.dims.map(String.init).joined(separator: ", ")))") · \(u.conditioning.candidateCount) candidates")
                        .foregroundStyle(.secondary)
                    if let a = u.aggregate {
                        Text(String(format: "Narrowness %.2f · mean effective rank %.1f · mean concentration %.2f", a.narrowness, a.meanEffectiveRank, a.meanConcentration))
                    }
                    ForEach(u.notes, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                }
            }

            ResultSection(title: "Metadata") {
                let md = report.metadata
                Text("Sources: \(md.sources.isEmpty ? "none" : md.sources.joined(separator: ", "))")
                if !md.trainingTags.isEmpty {
                    Text("Training tags: " + md.trainingTags.prefix(12).map { "\($0.tag) (\($0.count))" }.joined(separator: ", "))
                        .font(.caption).textSelection(.enabled)
                }
                if !md.triggerWords.isEmpty { Text("Triggers: \(md.triggerWords.joined(separator: ", "))").font(.caption) }
                if let m = report.metadataMatch, !m.matches.isEmpty {
                    Text("Matched: " + m.matches.prefix(6).map { String(format: "%@ ↔ %@ (%.2f)", $0.descriptor, $0.term, $0.similarity) }.joined(separator: ", "))
                        .font(.caption)
                }
            }

            if !report.reference.descriptors.isEmpty {
                ResultSection(title: "Reference description (CLIP)") {
                    Text(report.reference.descriptors.filter { $0.category != "tags" }
                        .map { String(format: "%@ %.2f", $0.phrase, $0.probability) }.joined(separator: " · "))
                        .font(.caption)
                }
            }

            ResultSection(title: "Probes") {
                Text(report.probes.note).font(.caption).foregroundStyle(.secondary)
                Text("Seed schedule \(report.reference.seedSchedule.id) · \(report.reference.permutations) permutations · \(report.reference.band.description)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Text(report.disclaimer).font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

struct ResearchView: View {
    let report: GMMResearchReport

    var body: some View {
        ResultSection(title: "Theory harness · analytic models (exact answers known)") {
            Text("Region \(report.regionLabel) covers \(String(format: "%.0f%%", report.regionAreaFraction * 100)) of the latent · \(report.options.permutations) seed permutations · \(report.options.band.description)")
                .font(.caption).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                GridRow {
                    Text("Model").bold()
                    Text("Exact Δbits").bold()
                    Text("Estimated Δbits (90% CI)").bold()
                    Text("Face share").bold()
                }
                ForEach(report.variants, id: \.name) { v in
                    GridRow {
                        Text(v.name)
                        Text(String(format: "%+.1f", v.exactDeltaBits)).monospacedDigit()
                        Text(String(format: "%+.1f  [%+.1f, %+.1f]", v.estimate.deltaBits, v.estimate.ci90[0], v.estimate.ci90[1])).monospacedDigit()
                        Text(v.region?.share.map { String(format: "%.0f%%", $0 * 100) } ?? "—").monospacedDigit()
                    }
                }
            }
            .font(.callout)
            if let c = report.crnVarianceRatio {
                Text(String(format: "Shared seeds cut estimator variance to %.1f%% of unpaired sampling.", c * 100)).font(.caption)
            }
            Text(String(format: "Seed contribution inside the face footprint: %.0f%% (separable) vs %.0f%% (coupled).",
                        report.sensitivitySeparable.insideMassFraction * 100, report.sensitivityCoupled.insideMassFraction * 100))
                .font(.caption)
        }
    }
}

struct NeedleView: View {
    let report: NeedleResearchReport

    var body: some View {
        ResultSection(title: "Face-seed needle · analytic models (true mass known)") {
            Text(String(format: "Seed segment for the face: %.0f%% of the seed · %.0f%% of its influence · true needle mass π = %.0e (%.1f bits of luck)",
                        report.footprint.fraction * 100, report.footprint.capturedInfluence * 100, report.truth, -log2(report.truth)))
                .font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 16) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("Criterion").bold()
                        Text("μ target").bold()
                        Text("μ base").bold()
                        Text("Δbits").bold()
                        Text("ι").bold()
                    }
                    ForEach([("Identity oracle", report.oracle), ("Latent fidelity", report.fidelity)], id: \.0) { label, r in
                        if let t = r.target.first, let b = r.base.first, let d = r.deltaBits.first {
                            GridRow {
                                Text(label)
                                Text(String(format: "%.1e", t.estimate.probability)).monospacedDigit()
                                Text((b.estimate.reachedTarget ? "" : "< ") + String(format: "%.1e", b.estimate.probability)).monospacedDigit()
                                Text((d.isLowerBound ? "≥ " : "") + String(format: "%+.1f", d.value)).monospacedDigit()
                                Text(t.backgroundIndependence.map { String(format: "%.2f", $0) } ?? "—").monospacedDigit()
                            }
                        }
                    }
                }
                .font(.callout)
                Spacer()
                FootprintHeatmap(values: report.footprintInfluence, shape: report.latentShape)
                    .frame(width: 96, height: 96)
                    .help("Where the face's seed contribution lives (noise space; brighter = more influence).")
            }
            if let curve = report.oracle.attackCurve.first(where: { $0.attempts == 1000 }) {
                Text(String(format: "An attacker drawing 1,000 seeds succeeds with probability %.2f on the target vs %@%.1e on the base.",
                            curve.target, curve.baseIsBound ? "≤ " : "", curve.base))
                    .font(.caption)
            }
            Text(String(format: "Latent fidelity accepts %.0f%% of true identity samples and %.2f%% of others (no face is ever decoded).",
                        report.agreement.tpr * 100, report.agreement.fpr * 100))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct MemoryView: View {
    let report: MemoryResearchReport

    var body: some View {
        ResultSection(title: "Memory pools · analytic models (memorization strength κ known)") {
            Text("Is your photo — or other photos of your face — stored in the weights? κ = 0: the face is learned; κ = 1: the training photos are stored. Statistics only; nothing is generated or decoded.")
                .font(.caption).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("κ").bold()
                    Text("Your photo in training").bold()
                    Text("Collapse").bold()
                    Text("Snap").bold()
                    Text("Seed radius T/B").bold()
                    Text("Your photo held out").bold()
                    Text("Recurring copies").bold()
                }
                ForEach(report.rows, id: \.kappa) { row in
                    GridRow {
                        Text(String(format: "%.2f", row.kappa)).monospacedDigit()
                        TierChip(tier: row.member.tier)
                        Text(String(format: "%.2f", row.member.peakSignal)).monospacedDigit()
                        Text(String(format: "%.0f×", row.member.peakSnap)).monospacedDigit()
                        Text(row.member.basin.map { String(format: "%.2f / %.2f", $0.halfRadiusTarget, $0.halfRadiusBase) } ?? "—")
                            .monospacedDigit()
                        TierChip(tier: row.heldOut.tier)
                        Text(row.heldOut.capture.first?.nearDuplicateRate.map { String(format: "%.0f%%", $0 * 100) } ?? "—")
                            .monospacedDigit()
                    }
                }
            }
            .font(.callout)
            if let strongest = report.rows.max(by: { $0.kappa < $1.kappa }) {
                let m = strongest.member
                HStack(alignment: .top, spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Retained variance on the face vs noise level (κ = \(String(format: "%.1f", strongest.kappa)), member)")
                            .font(.caption)
                        LineChart(series: [
                            (.orange, m.points.map { (Double($0.logSNR), $0.retainedTarget?.face ?? $0.spreadTarget.face) }),
                            (.gray, m.points.map { (Double($0.logSNR), $0.baseRetained) }),
                        ], yRange: 0...1.1)
                        .frame(width: 220, height: 110)
                        Text("orange: target · gray: base — the gap is the pool").font(.caption2).foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Collapse map").font(.caption)
                        FootprintHeatmap(values: m.map.map { max($0, 0) }, shape: m.mapShape)
                            .frame(width: 110, height: 110)
                            .help("Where the model pins the reference (latent grid; brighter = more collapse).")
                    }
                    if let basin = m.basin {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Seed endpoint basin r(β)").font(.caption)
                            LineChart(series: [
                                (.orange, zip(basin.betas, basin.target).map { (Double($0.0), $0.1) }),
                                (.gray, zip(basin.betas, basin.base).map { (Double($0.0), $0.1) }),
                            ], yRange: 0...1)
                            .frame(width: 220, height: 110)
                            Text("share of perturbed seeds returning to your photo").font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
                ForEach(m.reasons, id: \.self) { Text("· " + $0).font(.caption) }
            }
        }
    }
}

struct TierChip: View {
    let tier: MemoryPoolResult.Tier

    var body: some View {
        Text(tier.rawValue)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }

    private var color: Color {
        switch tier {
        case .memorizedPhoto: return .red
        case .identityPool: return .orange
        case .nearbyPool: return .yellow
        case .generalizedLikeness: return .blue
        case .noEvidence: return .secondary
        }
    }
}

/// Minimal line chart (Canvas): series of (x, y) points on a shared x range.
struct LineChart: View {
    let series: [(Color, [(Double, Double)])]
    let yRange: ClosedRange<Double>

    var body: some View {
        Canvas { ctx, size in
            let xs = series.flatMap { $0.1.map(\.0) }
            guard let lo = xs.min(), let hi = xs.max(), hi > lo else { return }
            func point(_ p: (Double, Double)) -> CGPoint {
                let y = min(max(p.1, yRange.lowerBound), yRange.upperBound)
                return CGPoint(x: (p.0 - lo) / (hi - lo) * size.width,
                               y: size.height - (y - yRange.lowerBound) / (yRange.upperBound - yRange.lowerBound) * size.height)
            }
            ctx.stroke(Path(CGRect(origin: .zero, size: size)), with: .color(.secondary.opacity(0.3)))
            for (color, points) in series where points.count > 1 {
                var path = Path()
                path.move(to: point(points[0]))
                for p in points.dropFirst() { path.addLine(to: point(p)) }
                ctx.stroke(path, with: .color(color), lineWidth: 2)
            }
        }
    }
}

struct FootprintHeatmap: View {
    let values: [Float]
    let shape: [Int]

    var body: some View {
        Canvas { ctx, size in
            guard shape.count >= 2 else { return }
            let h = shape[shape.count - 2], w = shape[shape.count - 1]
            let plane = Array(values.prefix(h * w))
            let peak = max(plane.max() ?? 1, 1e-9)
            let cw = size.width / CGFloat(w), ch = size.height / CGFloat(h)
            for y in 0..<h {
                for x in 0..<w {
                    let v = Double(plane[y * w + x] / peak)
                    ctx.fill(Path(CGRect(x: CGFloat(x) * cw, y: CGFloat(y) * ch, width: cw + 0.5, height: ch + 0.5)),
                             with: .color(Color(red: 0.15 + 0.85 * v, green: 0.2 + 0.5 * v, blue: 0.35 * (1 - v) + 0.2)))
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct ScoreRing: View {
    let value: Double

    var body: some View {
        ZStack {
            Circle().stroke(.quaternary, lineWidth: 10)
            Circle()
                .trim(from: 0, to: value)
                .stroke(color, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int((value * 100).rounded()))").font(.system(size: 28, weight: .semibold, design: .rounded)).monospacedDigit()
        }
        .frame(width: 84, height: 84)
    }

    private var color: Color {
        value >= 0.66 ? .red : value >= 0.4 ? .orange : .green
    }
}

struct ResultSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
