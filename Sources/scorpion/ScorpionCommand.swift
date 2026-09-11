//
//  ScorpionCommand.swift
//  scorpion
//
//  CLI for ScorpionKit: inspect a model link, describe a reference, scan the pair,
//  validate the theory on analytic models, manage the fingerprint bank, calibrate.
//

import ArgumentParser
import Foundation
import ScorpionKit

@main
struct ScorpionCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scorpion",
        abstract: "Estimate how likely a model is to reproduce a reference image's likeness — without downloading the whole model.",
        version: ScorpionInfo.version,
        subcommands: [Scan.self, Inspect.self, Describe.self, Research.self, Experiment.self, Bank.self, Eval.self],
        defaultSubcommand: Scan.self)
}

struct CommonOptions: ParsableArguments {
    @Option(help: "Maximum bytes of tensor data to fetch, e.g. 256MB, 1GB.")
    var budget: String = "256MB"

    @Option(help: "Number of seed permutations.")
    var seeds: Int = 64

    @Option(help: "Log-SNR band \"lo,hi\" (the probe's range of steps).")
    var band: String = "-4,2"

    @Flag(help: "Skip CLIP (no description; metadata matching disabled).")
    var noClip = false

    @Option(help: "CLIP checkpoint (HF repo with model.safetensors).")
    var clip: String = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    @Option(help: "Face to analyze when the reference has several (0 = largest).")
    var face: Int?

    @Option(help: "Calibration JSON from `scorpion eval`.")
    var calibration: String?

    @Option(help: "Fingerprint bank directory.")
    var bank: String?

    func scanOptions() throws -> ScanOptions {
        var o = ScanOptions()
        guard let bytes = parseBytes(budget) else { throw ValidationError("Bad --budget \(budget)") }
        o.budgetBytes = bytes
        o.permutations = seeds
        guard let b = LogSNRBand(parsing: band) else { throw ValidationError("Bad --band \(band); use lo,hi") }
        o.band = b
        o.useCLIP = !noClip
        o.clipRepo = clip
        o.faceIndex = face
        if let calibration { o.calibration = try Calibration.load(URL(fileURLWithPath: calibration)) }
        if let bank { o.bankDirectory = URL(fileURLWithPath: bank) }
        return o
    }
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
                var show = p.phase != lastPhase
                switch p.phase {
                case .fetchingTensors:
                    show = show || (!p.detail.isEmpty && p.detail != lastDetail) || p.bytesFetched - lastBytes >= 8 << 20
                case .loadingCLIP:
                    // Download percentages: print on each new tens digit.
                    show = show || (p.detail.hasSuffix("0%") && p.detail != lastDetail)
                default:
                    break
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

// MARK: - scan

struct Scan: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Scan a model link against a reference image.")

    @Option(name: .shortAndLong, help: "Model link (Hugging Face or GitHub).")
    var model: String

    @Option(name: .shortAndLong, help: "Reference image path.")
    var image: String

    @OptionGroup var common: CommonOptions

    @Option(help: "Write the JSON report to a path (or - for stdout).")
    var json: String?

    @Flag(name: .shortAndLong) var quiet = false

    func run() async throws {
        let scorpion = Scorpion(options: try common.scanOptions())
        let report = try await scorpion.scan(model: model, image: URL(fileURLWithPath: image),
                                             progress: ProgressPrinter(quiet: quiet).handler)
        if json != "-" { print(Render.report(report)) }
        try writeJSON(report, to: json)
    }
}

// MARK: - inspect

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Header-only inventory and fetch plan for a model link (fetches no weights).")

    @Option(name: .shortAndLong, help: "Model link (Hugging Face or GitHub).")
    var model: String

    @OptionGroup var common: CommonOptions

    @Option(help: "Write JSON to a path (or - for stdout).")
    var json: String?

    func run() async throws {
        let scorpion = Scorpion(options: try common.scanOptions())
        let inspection = try await scorpion.inspect(model: model, progress: ProgressPrinter(quiet: false).handler)
        if json != "-" { print(Render.inspection(inspection)) }
        try writeJSON(inspection, to: json)
    }
}

// MARK: - describe

struct Describe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Describe a reference image: content ID, faces, CLIP descriptors, seed permutations.")

    @Option(name: .shortAndLong, help: "Reference image path.")
    var image: String

    @OptionGroup var common: CommonOptions

    @Option(help: "How many seed permutations to print.")
    var show: Int = 8

    @Option(help: "Write JSON to a path (or - for stdout).")
    var json: String?

    @Option(help: "Write the selected face's segmentation mask (grayscale PNG, silhouette only).")
    var saveMask: String?

    func run() async throws {
        let scorpion = Scorpion(options: try common.scanOptions())
        let reference = try ReferenceImage.load(url: URL(fileURLWithPath: image))
        let analysis = try await scorpion.analyzeReference(reference, progress: ProgressPrinter(quiet: false).handler)
        if json != "-" { print(Render.reference(analysis, show: show)) }
        if let saveMask {
            guard let index = analysis.selectedFace else { throw ValidationError("No face detected; nothing to save.") }
            let segment = FaceSegmenter.segment(reference, face: analysis.faces[index])
            guard let raster = segment.face.rasterMask else { throw ValidationError("Landmarks unavailable; only an ellipse mask exists.") }
            try FaceSegmenter.writePNG(raster, to: URL(fileURLWithPath: saveMask))
            stderrPrint("wrote \(saveMask) (\(segment.method), \(raster.width)×\(raster.height))")
        }
        try writeJSON(analysis, to: json)
    }
}

// MARK: - experiment

struct Experiment: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run the face-seed needle experiment protocol.",
        subcommands: [Run.self])

    struct Run: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Score every (model, reference set) pair in a manifest and render the go/no-go verdict.",
            discussion: "See Docs/EXPERIMENT.md. A `toy` block synthesizes identities, look-alikes and models (dry run).")

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
    static let configuration = CommandConfiguration(abstract: "Validate the seed/band/region theory.",
                                                    subcommands: [GMM.self, Needle.self, Memory.self])

    struct Memory: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Memory pools: is your photo — or your face — stored in a model's weights? On analytic models with known memorization.",
            discussion: """
            For each memorization strength κ (0 = the identity is learned, 1 = the training photos are stored), \
            builds a fine-tune that saw your photo and one that saw only other photos of your face, and asks: is \
            there a pool at your face (curvature collapse), is it yours (where the model pulls your noised photo), \
            can a seed reach it (basin around your photo's seed endpoint), and does it take over other generations \
            (capture and recurring copies). Reference-directed and decode-free: statistics only.
            """)

        @Option(name: .shortAndLong, help: "Reference image; repeat to add more photos of the same person.")
        var image: [String]

        @Option(help: "Memorization strengths κ, e.g. \"0,0.5,0.9,1\".") var kappa: String = "0,0.5,0.9,1"
        @Option(help: "Training photos the fine-tune saw.") var members: Int = 12
        @Option(help: "Copies of the first training photo (duplication).") var duplicates: Int = 1
        @Option(help: "face (identity fine-tune) | photo (whole photo memorized).") var scope: String = "face"
        @Option(help: "Trigger word gating the memory (\"none\" = every prompt sees it).") var trigger: String = "sks person"
        @Option(help: "Identity weight the unconditional model keeps when gated.") var leak: Double = 1e-3
        @Option(help: "Identity weight under the trigger.") var identityWeight: Double = 0.05
        @Option(help: "Hutchinson probes (0 = forward-only).") var probes: Int = 16
        @Option(help: "Latent side length.") var side: Int = 16
        @Option(help: "References per kind in the membership audit.") var audit: Int = 6
        @Option(help: "Write JSON to a path (or - for stdout).") var json: String?

        func run() async throws {
            guard let first = image.first else { throw ValidationError("Pass at least one --image.") }
            var options = MemoryResearchOptions()
            options.kappas = kappa.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard !options.kappas.isEmpty, options.kappas.allSatisfy({ (0...1).contains($0) }) else {
                throw ValidationError("--kappa takes values in [0, 1]")
            }
            guard let s = MemoryScenario.Scope(rawValue: scope) else { throw ValidationError("Bad --scope") }
            options.scope = s
            options.members = max(members, 2)
            options.duplicates = max(duplicates, 1)
            options.trigger = trigger == "none" ? nil : trigger
            options.leak = leak
            options.identityWeight = identityWeight
            options.memory.probes = max(probes, 0)
            options.side = side
            options.audit = max(audit, 2)
            let reference = try ReferenceImage.load(url: URL(fileURLWithPath: first))
            let extra = try image.dropFirst().map { try ReferenceImage.load(url: URL(fileURLWithPath: $0)) }
            let report = try MemoryResearch.run(reference: reference, extraPhotos: extra, options: options) { stderrPrint("· \($0)") }
            if json != "-" { print(Render.memory(report)) }
            try writeJSON(report, to: json)
        }
    }

    struct Needle: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Face-seed needle on analytic models built around your photo, next to the true needle mass.")

        @Option(name: .shortAndLong, help: "Reference image; repeat to add more photos of the same person.")
        var image: [String]

        @Option(help: "Directory of control faces (look-alikes and unrelated people).")
        var controls: String?

        @Option(help: "True needle mass π: the identity's mixture weight in the target.") var needleWeight: Double = 1e-3
        @Option(help: "separable | coupled (face tied to background).") var coupling: String = "separable"
        @Option(help: "Guidance scales, e.g. \"1,3.5,7\" (meaningful with --trigger).") var guidance: String = "1"
        @Option(help: "Make the identity trigger-gated with this word.") var trigger: String?
        @Option(help: "Identity photo-to-photo spread (variance per latent coordinate).") var variance: Float = 0.05
        @Option(help: "Latent side length.") var side: Int = 16
        @Option(help: "Samples per subset-simulation level.") var samples: Int = 200
        @Option(help: "Sampler steps (the needle mass is defined for this sampler).") var steps: Int?
        @Option(help: "Write JSON to a path (or - for stdout).") var json: String?

        func run() async throws {
            guard let first = image.first else { throw ValidationError("Pass at least one --image.") }
            var options = NeedleResearchOptions()
            if let steps { options.needle.sampler.steps = steps }
            options.needleWeight = needleWeight
            guard let c = GMMScenario.Coupling(rawValue: coupling) else { throw ValidationError("Bad --coupling") }
            options.coupling = c
            options.guidance = guidance.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if options.guidance.isEmpty { options.guidance = [1] }
            options.trigger = trigger
            options.variance = variance
            options.side = side
            options.needle.subset.samplesPerLevel = samples
            let reference = try ReferenceImage.load(url: URL(fileURLWithPath: first))
            let extra = try image.dropFirst().map { try ReferenceImage.load(url: URL(fileURLWithPath: $0)) }
            var controlImages: [ReferenceImage] = []
            if let controls {
                let files = try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: controls), includingPropertiesForKeys: nil)
                controlImages = files.compactMap { try? ReferenceImage.load(url: $0) }
            }
            let report = try NeedleResearch.run(reference: reference, extraPhotos: extra, controls: controlImages, options: options)
            if json != "-" { print(Render.needle(report)) }
            try writeJSON(report, to: json)
        }
    }

    struct GMM: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "gmm",
            abstract: "Run the probes against analytic Gaussian-mixture models built around the reference, next to exact answers.")

        @Option(name: .shortAndLong, help: "Reference image path.")
        var image: String

        @Option(help: "A different image for the control model (default: synthetic).")
        var other: String?

        @Option(help: "Directory of images for base-model components (default: synthetic fields).")
        var corpus: String?

        @Option(help: "Seed permutations.") var seeds: Int = 256
        @Option(help: "Log-SNR band \"lo,hi\".") var band: String = "-12,12"
        @Option(help: "Latent side length.") var side: Int = 16
        @Option(help: "Base mixture components.") var components: Int = 32

        @Option(help: "Write JSON to a path (or - for stdout).")
        var json: String?

        func run() async throws {
            var options = GMMResearchOptions()
            options.permutations = seeds
            guard let b = LogSNRBand(parsing: band) else { throw ValidationError("Bad --band") }
            options.band = b
            options.side = side
            options.components = components
            let reference = try ReferenceImage.load(url: URL(fileURLWithPath: image))
            let otherImage = try other.map { try ReferenceImage.load(url: URL(fileURLWithPath: $0)) }
            var corpusImages: [ReferenceImage] = []
            if let corpus {
                let files = try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: corpus), includingPropertiesForKeys: nil)
                corpusImages = files.compactMap { try? ReferenceImage.load(url: $0) }
            }
            let report = try GMMResearch.run(reference: reference, other: otherImage, corpus: corpusImages, options: options)
            if json != "-" { print(Render.research(report)) }
            try writeJSON(report, to: json)
        }
    }
}

// MARK: - bank

struct Bank: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Manage the fingerprint bank.", subcommands: [Add.self, List.self, Remove.self])

    struct Add: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Fingerprint a model link into the bank under a label (e.g. identity, style, object).")

        @Option(name: .shortAndLong) var model: String
        @Option(name: .shortAndLong) var label: String
        @OptionGroup var common: CommonOptions

        func run() async throws {
            let scorpion = Scorpion(options: try common.scanOptions())
            let added = try await scorpion.addToBank(model: model, label: label, progress: ProgressPrinter(quiet: false).handler)
            if added.isEmpty { print("No adapter fingerprints extracted.") }
            for e in added { print("added \(e.id)  \(e.label)  family \(e.familyKey)  \(e.modules.count) modules  \(e.unit)") }
        }
    }

    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "List bank entries.")
        @Option var bank: String?

        func run() throws {
            let b = try FingerprintBank(directory: bank.map { URL(fileURLWithPath: $0) } ?? FingerprintBank.defaultDirectory)
            if b.entries.isEmpty { print("Bank is empty (\(b.directory.path)).") }
            for e in b.entries { print("\(e.id)  \(e.label.padding(toLength: 10, withPad: " ", startingAt: 0))  \(e.familyKey)  \(e.source)  \(e.unit)") }
        }
    }

    struct Remove: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Remove a bank entry.")
        @Argument var id: String
        @Option var bank: String?

        func run() throws {
            let b = try FingerprintBank(directory: bank.map { URL(fileURLWithPath: $0) } ?? FingerprintBank.defaultDirectory)
            try b.remove(id: id)
            print("removed \(id)")
        }
    }
}

// MARK: - eval

struct Eval: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Fit score calibration on labeled pairs.",
        discussion: """
        Manifest: JSON array of {"model": "<link>", "image": "<path>", "match": true|false}.
        Scans every pair, fits a logistic calibration over the score components, prints AUC,
        and writes the calibration JSON for `--calibration`.
        """)

    @Option(help: "Labeled manifest JSON.") var manifest: String
    @Option(help: "Output calibration path.") var output: String = "calibration.json"
    @OptionGroup var common: CommonOptions

    struct Pair: Decodable {
        let model: String
        let image: String
        let match: Bool
    }

    func run() async throws {
        let manifestURL = URL(fileURLWithPath: manifest)
        let pairs = try JSONDecoder().decode([Pair].self, from: Data(contentsOf: manifestURL))
        let scorpion = Scorpion(options: try common.scanOptions())
        var samples: [(components: [ScoreComponent], label: Bool)] = []
        for (i, pair) in pairs.enumerated() {
            let image = pair.image.hasPrefix("/") ? URL(fileURLWithPath: pair.image)
                : manifestURL.deletingLastPathComponent().appendingPathComponent(pair.image)
            do {
                let report = try await scorpion.scan(model: pair.model, image: image)
                samples.append((report.components, pair.match))
                stderrPrint(String(format: "[%d/%d] %.3f  %@  %@", i + 1, pairs.count, report.harmScore, pair.match ? "match" : "none", pair.model))
            } catch {
                stderrPrint("[\(i + 1)/\(pairs.count)] skipped \(pair.model): \(error.localizedDescription)")
            }
        }
        guard samples.contains(where: \.label), samples.contains(where: { !$0.label }) else {
            throw ValidationError("Need both matching and non-matching pairs to calibrate.")
        }
        let uncalibrated = samples.map { ScoreFusion.fuse($0.components, calibration: nil) }
        let before = ScoreFusion.auc(scores: uncalibrated, labels: samples.map(\.label))
        let calibration = ScoreFusion.fit(samples)
        print(String(format: "pairs: %d   AUC provisional: %@   AUC calibrated (in-sample): %@", samples.count,
                     before.map { String(format: "%.3f", $0) } ?? "n/a", calibration.auc.map { String(format: "%.3f", $0) } ?? "n/a"))
        try writeJSON(calibration, to: output)
    }
}
