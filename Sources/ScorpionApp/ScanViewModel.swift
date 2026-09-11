//
//  ScanViewModel.swift
//  ScorpionApp
//

import AppKit
import ScorpionKit
import UniformTypeIdentifiers

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var backend = "toy"
    @Published var modelLink = ""
    @Published var prompt = ""
    @Published var profile = "toy"
    @Published var heldOut = false
    @Published var runBasin = false
    @Published var image: NSImage?
    @Published var imageSize: CGSize = .zero
    @Published var controlsDirectory: URL?
    @Published var controlCount = 0

    @Published var isRunning = false
    @Published var phase = ""
    @Published var bytesFetched = 0
    @Published var weightBytes: Int?
    @Published var report: MemorizationReport? { didSet { refreshOverlay() } }
    @Published var layer = MemorizationHeatmap.primaryLayer { didSet { refreshOverlay() } }
    @Published var opacity = 1.0
    @Published var overlay: NSImage?
    @Published var research: MemoryResearchReport?
    @Published var errorMessage: String?

    private var reference: ReferenceImage?
    private var controls: [ReferenceImage] = []
    private var imageURL: URL?

    var backends: [String] { BackendRegistry.shared.names }
    var isToy: Bool { backend.hasPrefix("toy") }
    var canRun: Bool { !isRunning && reference != nil && (isToy || !modelLink.trimmingCharacters(in: .whitespaces).isEmpty) }
    var imageName: String? { imageURL?.lastPathComponent }
    var layers: [String] { report?.result.heatmap.layers.map(\.name) ?? [] }

    func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { loadImage(url) }
    }

    func loadImage(_ url: URL) {
        errorMessage = nil
        do {
            let ref = try ReferenceImage.load(url: url)
            reference = ref
            imageURL = url
            image = NSImage(cgImage: ref.image, size: NSSize(width: ref.width, height: ref.height))
            imageSize = CGSize(width: ref.width, height: ref.height)
            report = nil
            research = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func chooseControls() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Use as Controls"
        panel.message = "A folder of images known NOT to be in the model's training set."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "webp"]
        let files = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        controls = files.filter { exts.contains($0.pathExtension.lowercased()) }.sorted { $0.path < $1.path }
            .compactMap { try? ReferenceImage.load(url: $0) }
        controlsDirectory = url
        controlCount = controls.count
    }

    func clearControls() {
        controls = []
        controlsDirectory = nil
        controlCount = 0
    }

    func run() {
        guard canRun, let reference else { return }
        isRunning = true
        errorMessage = nil
        report = nil
        bytesFetched = 0
        weightBytes = nil
        phase = "Starting"
        var options: [String: [String]] = [:]
        if isToy && heldOut { options["member"] = ["false"] }
        let request = BackendRequest(backend: backend, model: isToy ? nil : modelLink.trimmingCharacters(in: .whitespaces),
                                     reference: reference, prompt: prompt, options: options)
        let config = MemorizationConfiguration.named(profile) ?? (isToy ? .toy : .standard)
        let basin: SeedBasinConfiguration? = runBasin ? (isToy ? SeedBasinConfiguration() : .lite) : nil
        let controls = controls
        let scorpion = Scorpion(options: ScorpionOptions())
        Task.detached(priority: .userInitiated) {
            do {
                let result = try await scorpion.detectMemorization(request, controls: controls, configuration: config, basin: basin) { p in
                    Task { @MainActor in self.apply(p) }
                }
                await MainActor.run {
                    self.report = result
                    self.isRunning = false
                    self.phase = "Done — \(result.result.verdict.rawValue)"
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isRunning = false
                    self.phase = ""
                }
            }
        }
    }

    func runResearch() {
        guard !isRunning, let reference else { return }
        isRunning = true
        errorMessage = nil
        phase = "Sweeping memorization strength on analytic worlds"
        Task.detached(priority: .userInitiated) {
            do {
                var options = MemoryResearchOptions()
                options.kappas = [0, 0.5, 1]
                options.audit = 4
                let memory = try MemoryResearch.run(reference: reference, options: options)
                await MainActor.run {
                    self.research = memory
                    self.isRunning = false
                    self.phase = "Done"
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isRunning = false
                }
            }
        }
    }

    private func refreshOverlay() {
        guard let heatmap = report?.result.heatmap, heatmap.layer(layer) != nil else {
            overlay = nil
            return
        }
        var renderer = HeatmapRenderer()
        renderer.layer = layer
        overlay = renderer.overlay(heatmap, maxSide: 640).map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
    }

    private func apply(_ p: ScanProgress) {
        guard isRunning else { return }
        phase = p.detail.isEmpty ? p.phase.rawValue : "\(p.phase.rawValue) — \(p.detail)"
        bytesFetched = p.bytesFetched
        if let w = p.weightBytes { weightBytes = w }
    }

    func exportReport() {
        guard let report else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "scorpion-memorization-\(report.reference.id.prefix(8)).json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
            try encoder.encode(report).write(to: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func exportOverlay() {
        guard let report, let reference else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "scorpion-heatmap-\(report.reference.id.prefix(8)).png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var renderer = HeatmapRenderer()
        renderer.layer = layer
        let r = report.result
        guard let img = renderer.render(r.heatmap, reference: reference, title: "\(r.verdict.rawValue) · \(report.backend)",
                                        subtitle: "\(reference.name) · \(r.estimator) · uncalibrated") else { return }
        do {
            try ImageWriter.writePNG(img, to: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
