//
//  ScanViewModel.swift
//  ScorpionApp
//

import AppKit
import ScorpionKit
import UniformTypeIdentifiers

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var modelLink = ""
    @Published var image: NSImage?
    @Published var imageSize: CGSize = .zero
    @Published var faces: [FaceRegion] = []
    @Published var selectedFace = 0
    @Published var budgetMB: Double = 256
    @Published var useCLIP = true

    @Published var isRunning = false
    @Published var phase = ""
    @Published var bytesFetched = 0
    @Published var weightBytes: Int?
    @Published var report: LikenessReport?
    @Published var research: GMMResearchReport?
    @Published var needle: NeedleResearchReport?
    @Published var memory: MemoryResearchReport?
    @Published var errorMessage: String?
    /// Tinted segmentation masks per face index (overlays on the reference).
    @Published var segmentOverlays: [Int: NSImage] = [:]
    @Published var segmentMethods: [Int: String] = [:]

    private var reference: ReferenceImage?
    private var imageURL: URL?

    var canScan: Bool { !isRunning && reference != nil && !modelLink.trimmingCharacters(in: .whitespaces).isEmpty }
    var imageName: String? { imageURL?.lastPathComponent }

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
            faces = []
            selectedFace = 0
            report = nil
            research = nil
            needle = nil
            memory = nil
            segmentOverlays = [:]
            segmentMethods = [:]
            Task.detached(priority: .userInitiated) {
                let found = (try? FaceDetector.detect(ref)) ?? []
                let segments = found.map { FaceSegmenter.segment(ref, face: $0) }
                let overlays = segments.reduce(into: [Int: NSImage]()) { acc, seg in
                    if let raster = seg.face.rasterMask { acc[seg.faceIndex] = Self.tint(raster) }
                }
                let methods = segments.reduce(into: [Int: String]()) { $0[$1.faceIndex] = $1.method }
                await MainActor.run {
                    self.faces = found
                    self.segmentOverlays = overlays
                    self.segmentMethods = methods
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func options() -> ScanOptions {
        var o = ScanOptions()
        o.budgetBytes = Int(budgetMB) * 1_000_000
        o.useCLIP = useCLIP
        o.faceIndex = faces.isEmpty ? nil : selectedFace
        return o
    }

    func scan() {
        guard canScan, let reference else { return }
        isRunning = true
        errorMessage = nil
        report = nil
        bytesFetched = 0
        weightBytes = nil
        phase = "Starting"
        let link = modelLink.trimmingCharacters(in: .whitespaces)
        let scorpion = Scorpion(options: options())
        Task.detached(priority: .userInitiated) {
            do {
                let result = try await scorpion.scan(model: link, reference: reference) { p in
                    Task { @MainActor in self.apply(p) }
                }
                await MainActor.run {
                    self.report = result
                    self.bytesFetched = result.source.bytesFetched
                    self.weightBytes = result.source.weightBytes
                    self.isRunning = false
                    self.phase = "Done"
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

    /// Mask raster → accent-tinted translucent image for overlaying.
    nonisolated static func tint(_ raster: RasterMask) -> NSImage {
        var rgba = [UInt8](repeating: 0, count: raster.width * raster.height * 4)
        for (i, v) in raster.values.enumerated() {
            let a = UInt8(max(0, min(255, v * 110)))
            rgba[4 * i] = UInt8(Int(40) * Int(a) / 255)
            rgba[4 * i + 1] = UInt8(Int(170) * Int(a) / 255)
            rgba[4 * i + 2] = UInt8(Int(255) * Int(a) / 255)
            rgba[4 * i + 3] = a
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let cg = CGImage(width: raster.width, height: raster.height, bitsPerComponent: 8, bitsPerPixel: 32,
                         bytesPerRow: raster.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
        return NSImage(cgImage: cg, size: NSSize(width: raster.width, height: raster.height))
    }

    func runResearch() {
        guard !isRunning, let reference else { return }
        isRunning = true
        errorMessage = nil
        phase = "Running theory harness (analytic models)"
        Task.detached(priority: .userInitiated) {
            do {
                let result = try GMMResearch.run(reference: reference)
                await MainActor.run { self.research = result; self.phase = "Running face-seed needle (analytic models)" }
                let needle = try NeedleResearch.run(reference: reference)
                await MainActor.run { self.needle = needle; self.phase = "Running memory pools (analytic models)" }
                // Learned vs stored, lighter than the CLI default (which sweeps κ = 0, 0.5, 0.9, 1).
                var options = MemoryResearchOptions()
                options.kappas = [0, 1]
                options.audit = 4
                let memory = try MemoryResearch.run(reference: reference, options: options)
                await MainActor.run {
                    self.memory = memory
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
        panel.nameFieldStringValue = "scorpion-report-\(report.reference.referenceID.prefix(8)).json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try report.jsonData().write(to: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
