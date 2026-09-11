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
                    .frame(minWidth: 340, idealWidth: 420, maxWidth: 520)
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
            Picker("", selection: $model.backend) {
                ForEach(model.backends, id: \.self) { Text($0).tag($0) }
            }
            .frame(width: 130)
            .onChange(of: model.backend) { _, new in model.profile = new.hasPrefix("toy") ? "toy" : "quick" }
            TextField(model.isToy ? "Toy world: memorization planted around your image" : "Model under test — LoRA link or path",
                      text: $model.modelLink)
                .textFieldStyle(.roundedBorder)
                .disabled(model.isToy)
                .onSubmit { model.run() }
            TextField("Prompt (default: trigger word)", text: $model.prompt)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 220)
            Button {
                model.run()
            } label: {
                Label("Test", systemImage: "play.fill").frame(minWidth: 64)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.canRun)
        }
        .padding(12)
    }

    // MARK: Reference

    private var referencePane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Reference").font(.headline)
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                        .foregroundStyle(.tertiary)
                    if let image = model.image {
                        HeatmapOverlayImage(image: image, imageSize: model.imageSize, heatmap: model.report?.result.heatmap,
                                            overlay: model.overlay, opacity: model.opacity)
                            .padding(8)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.badge.plus").font(.system(size: 34)).foregroundStyle(.secondary)
                            Text("Drop an image or choose one").foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(minHeight: 300)
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

                if model.report != nil {
                    GroupBox("Heatmap") {
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("Layer", selection: $model.layer) {
                                ForEach(model.layers, id: \.self) { Text($0).tag($0) }
                            }
                            HStack {
                                Text("Opacity")
                                Slider(value: $model.opacity, in: 0...1)
                            }
                            HeatmapLegend(threshold: model.report?.result.heatmap.threshold ?? 0.1)
                        }
                    }
                }

                GroupBox("Test settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Profile", selection: $model.profile) {
                            ForEach(["toy", "quick", "standard", "evidence"], id: \.self) { Text($0).tag($0) }
                        }
                        HStack {
                            Button("Controls…") { model.chooseControls() }
                            if model.controlCount > 0 {
                                Text("\(model.controlCount) images").foregroundStyle(.secondary)
                                Button("Clear") { model.clearControls() }.buttonStyle(.link)
                            } else {
                                Text("none — regions uncalibrated").foregroundStyle(.secondary)
                            }
                        }
                        .font(.callout)
                        if model.isToy {
                            Toggle("Reference held out of training", isOn: $model.heldOut)
                        }
                        Toggle("Seed-endpoint basin (slow)", isOn: $model.runBasin)
                    }
                }

                Button {
                    model.runResearch()
                } label: {
                    Label("Sweep memorization strength", systemImage: "function")
                }
                .disabled(model.image == nil || model.isRunning)
                .help("The probe on analytic worlds built around this image, at κ = 0, 0.5, 1, next to the exact collapse.")
            }
            .padding(12)
        }
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
                    ReportView(report: report, exportReport: model.exportReport, exportOverlay: model.exportOverlay)
                } else if model.isRunning {
                    VStack(spacing: 10) {
                        ProgressView()
                        Text(model.phase).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                } else if model.research == nil {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Has this model memorized regions of your image?").font(.title3)
                        Text("Scorpion measures where the model pins your image while its baseline leaves it free — coordinate-wise variance collapse at the noised image — and draws it as a heatmap over your picture. Nothing is generated for display and no latent is decoded.")
                            .foregroundStyle(.secondary)
                        Text("The toy backend plants memorization in known regions of your image, so you can see what a positive result looks like.")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: 560, alignment: .leading)
                    .padding(.top, 40)
                }
                if let research = model.research {
                    ResearchView(report: research)
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
            } else {
                Text("\(ByteFormat.string(fetched)) fetched")
            }
        }
    }
}

/// The reference, aspect-fit, with the heatmap overlay placed over the frame it covers and
/// counted regions outlined and numbered.
struct HeatmapOverlayImage: View {
    let image: NSImage
    let imageSize: CGSize
    let heatmap: MemorizationHeatmap?
    let overlay: NSImage?
    let opacity: Double

    var body: some View {
        GeometryReader { geo in
            let scale = min(geo.size.width / max(imageSize.width, 1), geo.size.height / max(imageSize.height, 1))
            let w = imageSize.width * scale, h = imageSize.height * scale
            let ox = (geo.size.width - w) / 2, oy = (geo.size.height - h) / 2
            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .resizable()
                    .frame(width: w, height: h)
                    .saturation(heatmap == nil ? 1 : 0.15)
                    .opacity(heatmap == nil ? 1 : 0.85)
                    .offset(x: ox, y: oy)
                if let heatmap, let overlay {
                    let f = heatmap.frame
                    Image(nsImage: overlay)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: f.width * scale, height: f.height * scale)
                        .opacity(opacity)
                        .offset(x: ox + f.minX * scale, y: oy + f.minY * scale)
                        .allowsHitTesting(false)
                    RegionOutlines(heatmap: heatmap, scale: scale)
                        .offset(x: ox, y: oy)
                        .allowsHitTesting(false)
                }
            }
        }
    }
}

struct RegionOutlines: View {
    let heatmap: MemorizationHeatmap
    let scale: CGFloat

    var body: some View {
        Canvas { ctx, _ in
            let g = heatmap.grid
            for region in heatmap.regions where region.counts {
                let cells = Set(region.cells)
                var path = Path()
                func pt(_ c: Int, _ r: Int) -> CGPoint {
                    let rect = heatmap.cellRect(row: 0, col: 0)
                    return CGPoint(x: (heatmap.frame.minX + CGFloat(c) * rect.width) * scale,
                                   y: (heatmap.frame.minY + CGFloat(r) * rect.height) * scale)
                }
                for cell in region.cells {
                    let r = cell / g.cols, c = cell % g.cols
                    if r == 0 || !cells.contains(cell - g.cols) { path.move(to: pt(c, r)); path.addLine(to: pt(c + 1, r)) }
                    if r == g.rows - 1 || !cells.contains(cell + g.cols) { path.move(to: pt(c, r + 1)); path.addLine(to: pt(c + 1, r + 1)) }
                    if c == 0 || !cells.contains(cell - 1) { path.move(to: pt(c, r)); path.addLine(to: pt(c, r + 1)) }
                    if c == g.cols - 1 || !cells.contains(cell + 1) { path.move(to: pt(c + 1, r)); path.addLine(to: pt(c + 1, r + 1)) }
                }
                ctx.stroke(path, with: .color(.white.opacity(0.9)), lineWidth: 3)
                ctx.stroke(path, with: .color(.black), lineWidth: 1.2)
                let first = region.cells.min() ?? 0
                let origin = pt(first % g.cols, first / g.cols)
                ctx.draw(Text("\(region.id)").font(.caption2.bold()).foregroundColor(.black),
                         at: CGPoint(x: origin.x + 8, y: origin.y + 8))
            }
        }
    }
}

struct HeatmapLegend: View {
    let threshold: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            LinearGradient(colors: stride(from: 0, through: 1, by: 0.125).map { t in
                let c = Colormap.memorization.color(Float(t))
                return Color(red: Double(c.r), green: Double(c.g), blue: Double(c.b))
            }, startPoint: .leading, endPoint: .trailing)
                .frame(height: 10)
                .clipShape(RoundedRectangle(cornerRadius: 2))
            HStack {
                Text(String(format: "%.2f τ", threshold))
                Spacer()
                Text("0.5")
                Spacer()
                Text("1 pinned")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            Text("Collapse ρ_base − ρ_target, absolute scale; clear below τ")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

struct ReportView: View {
    let report: MemorizationReport
    let exportReport: () -> Void
    let exportOverlay: () -> Void

    var body: some View {
        let r = report.result
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Label(title(r.verdict), systemImage: icon(r.verdict)).font(.title2.bold())
                Text("rule-based · uncalibrated").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Export PNG…", action: exportOverlay)
                Button("Export JSON…", action: exportReport)
            }
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(r.reasons.enumerated()), id: \.offset) { _, reason in
                    Text("· \(reason)").foregroundStyle(.secondary).textSelection(.enabled)
                }
            }

            if !r.regions.isEmpty {
                GroupBox("Regions") {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                        GridRow {
                            ForEach(["#", "area", "mean", "peak", "λ", "mass", "p", "status", "snap"], id: \.self) {
                                Text($0).font(.caption.bold()).foregroundStyle(.secondary)
                            }
                        }
                        ForEach(r.regions) { g in
                            GridRow {
                                Text("\(g.id)")
                                Text(String(format: "%.1f%%", 100 * g.areaFraction))
                                Text(String(format: "%.2f", g.meanCollapse))
                                Text(String(format: "%.2f", g.peakCollapse))
                                Text(String(format: "%g", g.peakLogSNR))
                                Text(String(format: "%.1f", g.mass))
                                Text(g.pValue.map { String(format: "%.3f", $0) } ?? "—")
                                Text(g.significance.rawValue).foregroundStyle(g.counts ? .primary : .secondary)
                                Text(String(format: "%.1f×", g.snap))
                            }
                            .font(.callout.monospacedDigit())
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            GroupBox("Noise levels") {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 3) {
                    GridRow {
                        ForEach(["λ", "released", "mean collapse", "max", "prompt", "frame snap"], id: \.self) {
                            Text($0).font(.caption.bold()).foregroundStyle(.secondary)
                        }
                    }
                    ForEach(r.levels, id: \.logSNR) { l in
                        GridRow {
                            Text(String(format: "%g", l.logSNR))
                            Text(String(format: "%.0f%%", 100 * l.releasedFraction))
                            Text(String(format: "%.3f", l.meanCollapse))
                            Text(String(format: "%.3f", l.maxCollapse))
                            Text(l.meanPromptCollapse.map { String(format: "%.3f", $0) } ?? "—")
                            Text(String(format: "%.2f", l.snap))
                        }
                        .font(.callout.monospacedDigit())
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let b = report.basin {
                GroupBox("Seed-endpoint basin (region \(b.region))") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(String(format: "β½ target %.2f vs base %.2f · %@", b.halfRadiusTarget, b.halfRadiusBase, b.sampler))
                        Text("r(β) target: " + b.target.map { String(format: "%.2f", $0) }.joined(separator: " "))
                            .font(.callout.monospacedDigit())
                        Text("r(β) base:   " + b.base.map { String(format: "%.2f", $0) }.joined(separator: " "))
                            .font(.callout.monospacedDigit())
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            GroupBox("Provenance") {
                VStack(alignment: .leading, spacing: 3) {
                    Text("reference sha256 \(report.reference.id)").textSelection(.enabled)
                    Text("\(report.backend) · target \(r.target) · base \(r.base) · prompt \(r.prompt.isConditional ? "\"\(r.prompt.text)\"" : "(unconditional)")")
                    ForEach(report.provenance.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in Text("\(k): \(v)") }
                    Text("\(r.estimator) · \(r.configuration.draws) draws · \(r.evaluations) evaluations · \(String(format: "%.1f", r.seconds)) s · schedule \(r.schedule.id)")
                    if let c = r.controls { Text("\(c.count) controls (\(c.source))") }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(report.disclaimer).font(.caption).foregroundStyle(.secondary)
        }
    }

    func title(_ v: MemorizationVerdict) -> String {
        switch v {
        case .memorized: return "Memorized"
        case .memorizedOtherContent: return "Memorized — other content"
        case .noEvidence: return "No evidence of memorization"
        case .inconclusive: return "Inconclusive"
        }
    }

    func icon(_ v: MemorizationVerdict) -> String {
        switch v {
        case .memorized: return "exclamationmark.octagon.fill"
        case .memorizedOtherContent: return "exclamationmark.triangle.fill"
        case .noEvidence: return "checkmark.circle"
        case .inconclusive: return "questionmark.circle"
        }
    }
}

struct ResearchView: View {
    let report: MemoryResearchReport

    var body: some View {
        GroupBox("Memorization strength sweep (analytic worlds)") {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                GridRow {
                    ForEach(["κ", "member verdict", "IoU", "exact", "estimated", "held-out verdict", "membership AUC"], id: \.self) {
                        Text($0).font(.caption.bold()).foregroundStyle(.secondary)
                    }
                }
                ForEach(report.rows, id: \.kappa) { row in
                    GridRow {
                        Text(String(format: "%.2f", row.kappa))
                        Text(row.member.verdict.rawValue)
                        Text(String(format: "%.2f", row.iou))
                        Text(String(format: "%.2f", row.exactPeakCollapse))
                        Text(row.estimatedPeakCollapse.map { String(format: "%.2f", $0) } ?? "—")
                        Text(row.heldOut.verdict.rawValue)
                        Text(row.membershipAUC.map { String(format: "%.2f", $0) } ?? "—")
                    }
                    .font(.callout.monospacedDigit())
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
