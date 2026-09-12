//
//  ScorpionCommand.swift
//  scorpion
//
//  CLI for ScorpionKit: test whether a model has memorized regions of a reference image
//  (heatmap overlay + report), inspect a model link, and validate the probe on analytic
//  worlds with known memorization.
//

import ArgumentParser
import CoreGraphics
import Foundation
import ScorpionKit

@main
struct ScorpionCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scorpion",
        abstract: "Test whether a diffusion model has memorized regions of a reference image — shown as a heatmap over the image.",
        version: ScorpionInfo.version,
        subcommands: [Memorization.self, Inspect.self, Research.self, Experiment.self],
        defaultSubcommand: Memorization.self)
}

/// Decimal units (1 MB = 10⁶ bytes), matching how sizes are reported; GiB/MiB/KiB are binary.
func parseBytes(_ s: String) -> Int? {
    let t = s.uppercased().replacingOccurrences(of: " ", with: "")
    let units: [(String, Int)] = [("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
                                  ("GB", 1_000_000_000), ("G", 1_000_000_000), ("MB", 1_000_000), ("M", 1_000_000),
                                  ("KB", 1_000), ("K", 1_000), ("B", 1)]
    for (suffix, mult) in units where t.hasSuffix(suffix) {
        return Double(t.dropLast(suffix.count)).map { Int($0 * Double(mult)) }
    }
    return Int(t)
}

func writeJSON<T: Encodable>(_ value: T, to path: String?) throws {
    guard let path else { return }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    let data = try encoder.encode(value)
    if path == "-" {
        FileHandle.standardOutput.write(data)
        print()
    } else {
        try data.write(to: URL(fileURLWithPath: path))
        stderrPrint("wrote \(path)")
    }
}

func stderrPrint(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

/// Progress on stderr, one line per phase (plus byte counts while fetching).
final class ProgressPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastPhase: ScanPhase?
    private var lastDetail = ""
    private var lastBytes = 0
    let quiet: Bool

    init(quiet: Bool) { self.quiet = quiet }

    var handler: ProgressHandler {
        { [self] p in
            guard !quiet else { return }
            lock.withLock {
                var show = p.phase != lastPhase || (p.detail != lastDetail && p.phase != .fetchingTensors)
                if p.phase == .fetchingTensors {
                    show = show || (!p.detail.isEmpty && p.detail != lastDetail) || p.bytesFetched - lastBytes >= 8 << 20
                }
                guard show else { return }
                lastPhase = p.phase
                if !p.detail.isEmpty { lastDetail = p.detail }
                lastBytes = p.bytesFetched
                var line = "· \(p.phase.rawValue)"
                if !p.detail.isEmpty { line += " — \(p.detail)" }
                if p.phase == .fetchingTensors { line += " [\(ByteFormat.string(p.bytesFetched)) fetched]" }
                stderrPrint(line)
            }
        }
    }
}

func loadImages(inDirectory path: String?) throws -> [ReferenceImage] {
    guard let path else { return [] }
    let url = URL(fileURLWithPath: path)
    let exts: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "webp", "bmp", "gif"]
    let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        .filter { exts.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    return try files.map { try ReferenceImage.load(url: $0) }
}

// MARK: - memorization

struct Memorization: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Has this model memorized regions of this image? Writes a heatmap overlay and a report.",
        discussion: """
        Measures coordinate-wise variance collapse (Kim & Lee, arXiv 2605.26756) at the noised reference: \
        where the model under test pins the image while its baseline leaves it free. Regions are tested against \
        negative controls (--controls: images known not to be in training) and confirmed on independent draws. \
        Nothing is generated for display and no latent is decoded; the overlay is drawn on your own image.

        Backends: `toy` builds an analytic world around the image with memorization planted in known regions \
        (--plant x,y,w,h:κ, repeatable). Real executors (e.g. flux2-klein) take --model and --option key=value.
        """)

    @Option(name: .shortAndLong, help: "Reference image.") var image: String
    @Option(help: "Backend: \(BackendRegistry.shared.names.joined(separator: ", ")) (default toy).") var backend: String = "toy"
    @Option(name: .shortAndLong, help: "Model under test (link or path), for real executors.") var model: String?
    @Option(help: "Prompt to condition on (default: the model's trigger word, else unconditional).") var prompt: String?
    @Option(help: "Directory of negative-control images (known non-members).") var controls: String?
    @Option(help: "quick | standard | evidence | toy (default: toy for the toy backend, else standard).") var profile: String?
    @Option(help: "Jacobian estimator: hutchinson-reverse | hutchinson-forward | exact.") var estimator: String?
    @Option(help: "Base-model latent samples to use as controls when --controls is not given.") var baseControls: Int?
    @Flag(help: "Also run the seed-endpoint basin on the strongest region (slow on real models).") var basin = false
    @Flag(help: "Use the full seed-basin settings (8 β × 32 trials × 128 steps) instead of lite.") var fullBasin = false
    @Option(help: "Toy: planted region x,y,w,h[:κ] in normalized frame coordinates (repeatable).") var plant: [String] = []
    @Flag(help: "Toy: the reference was held out of training (only other photos of it were seen).") var heldOut = false
    @Option(help: "Toy: grid side (default 32).") var side: Int?
    @Option(help: "Backend option key=value (repeatable), e.g. base-dir=/path.") var option: [String] = []
    @Option(help: "Heatmap layer to draw: collapse | collapse-raw | prompt-collapse | score-difference.") var layer = "collapse"
    @Option(help: "Write the heatmap overlay PNG here.") var overlay: String?
    @Option(help: "Write the JSON report to a path (or - for stdout).") var json: String?
    @Flag(name: .shortAndLong, help: "No progress output.") var quiet = false

    func run() async throws {
        Backends.registerAll()
        let reference = try ReferenceImage.load(url: URL(fileURLWithPath: image))
        var options: [String: [String]] = [:]
        for kv in option {
            let parts = kv.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { throw ValidationError("--option takes key=value (got \(kv))") }
            options[parts[0], default: []].append(parts[1])
        }
        if !plant.isEmpty { options["plant"] = plant }
        if heldOut { options["member"] = ["false"] }
        if let side { options["side"] = [String(side)] }

        let isToy = backend.hasPrefix("toy")
        guard var config = MemorizationConfiguration.named(profile ?? (isToy ? "toy" : "standard")) else {
            throw ValidationError("Unknown --profile \(profile ?? "")")
        }
        if let estimator {
            guard let kind = EstimatorKind(rawValue: estimator) else { throw ValidationError("Unknown --estimator \(estimator)") }
            config.estimator = kind
        }
        if let baseControls { config.baseSampleControls = max(baseControls, 0) }
        let controlImages = try loadImages(inDirectory: controls)
        let basinConfig: SeedBasinConfiguration? = basin ? (fullBasin || isToy ? SeedBasinConfiguration() : .lite) : nil

        let scorpion = Scorpion(options: ScorpionOptions())
        let request = BackendRequest(backend: backend, model: model, reference: reference, prompt: prompt ?? "", options: options)
        let report = try await scorpion.detectMemorization(request, controls: controlImages, configuration: config,
                                                           basin: basinConfig, progress: ProgressPrinter(quiet: quiet).handler)
        if json != "-" { print(Render.memorization(report)) }
        if let overlay {
            var renderer = HeatmapRenderer()
            renderer.layer = layer
            guard report.result.heatmap.layer(layer) != nil else { throw ValidationError("This report has no layer '\(layer)'.") }
            let r = report.result
            let title = "\(Render.verdictTitle(r.verdict)) · \(report.backend)\(report.model.map { " · \($0)" } ?? "")"
            let subtitle = "\(reference.name) · \(r.regions.filter(\.counts).count) region(s) · λ ∈ {\(r.scout.selected.map { String(format: "%g", $0) }.joined(separator: ", "))} · \(r.estimator) · uncalibrated"
            guard let img = renderer.render(r.heatmap, reference: reference, title: title, subtitle: subtitle) else {
                throw ValidationError("Could not render the overlay.")
            }
            try ImageWriter.writePNG(img, to: URL(fileURLWithPath: overlay))
            stderrPrint("wrote \(overlay)")
        }
        try writeJSON(report, to: json)
    }
}

// MARK: - inspect

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Header-only inventory and fetch plan for a model link (fetches no weights).")

    @Option(name: .shortAndLong, help: "Model link (Hugging Face or GitHub).") var model: String
    @Option(help: "Maximum bytes of tensor data a plan may cover, e.g. 256MB, 1GB.") var budget: String = "256MB"
    @Option(help: "Write JSON to a path (or - for stdout).") var json: String?

    func run() async throws {
        var o = ScorpionOptions()
        guard let bytes = parseBytes(budget) else { throw ValidationError("Bad --budget \(budget)") }
        o.budgetBytes = bytes
        let inspection = try await Scorpion(options: o).inspect(model: model, progress: ProgressPrinter(quiet: false).handler)
        if json != "-" { print(Render.inspection(inspection)) }
        try writeJSON(inspection, to: json)
    }
}

// MARK: - experiment

struct Experiment: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run the memorization go/no-go protocol.", subcommands: [Run.self])

    struct Run: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Probe every member and held-out reference in a manifest and render the go/no-go verdict.",
            discussion: "See Docs/EXPERIMENT.md. A `toy` block synthesizes subjects fine-tuned at several memorization strengths.")

        @Option(help: "Experiment manifest JSON.") var manifest: String
        @Option(help: "Write JSON to a path (or - for stdout).") var json: String?

        func run() async throws {
            let spec = try JSONDecoder().decode(ExperimentManifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifest)))
            let report = try ExperimentRunner.run(spec) { stderrPrint("· \($0)") }
            if json != "-" { print(Render.experiment(report)) }
            try writeJSON(report, to: json)
        }
    }
}

// MARK: - research

struct Research: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Validate the probe where the answer is known.", subcommands: [Memory.self])

    struct Memory: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Sweep memorization strength κ on analytic worlds built around your image, next to the exact collapse.",
            discussion: """
            For each κ (0 = the content is learned, 1 = the training photos are stored), plants a memorized region, \
            builds a fine-tune that saw your image and one that saw only other photos of it, runs the probe on both, \
            and audits members, held-out photos, look-alikes and strangers for the membership AUC.
            """)

        @Option(name: .shortAndLong, help: "Reference image.") var image: String
        @Option(help: "Memorization strengths κ, e.g. \"0,0.5,0.9,1\".") var kappa: String = "0,0.5,0.9,1"
        @Option(help: "Planted region x,y,w,h in normalized frame coordinates.") var plant: String = "0.25,0.25,0.5,0.5"
        @Option(help: "Training photos the fine-tune saw.") var members: Int = 12
        @Option(help: "Copies of the first training photo (duplication).") var duplicates: Int = 1
        @Option(help: "Trigger word gating the memory (\"none\" = every prompt sees it).") var trigger: String = "sks person"
        @Option(help: "Weight the unconditional model keeps when gated.") var leak: Double = 1e-3
        @Option(help: "Subject weight under the trigger.") var identityWeight: Double = 0.05
        @Option(help: "Hutchinson probes.") var probes: Int = 16
        @Option(help: "Grid side.") var side: Int = 16
        @Option(help: "References per kind in the membership audit.") var audit: Int = 6
        @Option(help: "Write JSON to a path (or - for stdout).") var json: String?

        func run() async throws {
            var options = MemoryResearchOptions()
            options.kappas = kappa.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard !options.kappas.isEmpty, options.kappas.allSatisfy({ (0...1).contains($0) }) else {
                throw ValidationError("--kappa takes values in [0, 1]")
            }
            let box = plant.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard box.count == 4, box.allSatisfy({ (0...1).contains($0) }) else { throw ValidationError("--plant takes x,y,w,h in [0, 1]") }
            options.plant = box
            options.members = max(members, 2)
            options.duplicates = max(duplicates, 1)
            options.trigger = trigger == "none" ? nil : trigger
            options.leak = leak
            options.identityWeight = identityWeight
            options.configuration.probes = max(probes, 1)
            options.side = side
            options.audit = max(audit, 2)
            let reference = try ReferenceImage.load(url: URL(fileURLWithPath: image))
            let report = try MemoryResearch.run(reference: reference, options: options) { stderrPrint("· \($0)") }
            if json != "-" { print(Render.memoryResearch(report)) }
            try writeJSON(report, to: json)
        }
    }
}
