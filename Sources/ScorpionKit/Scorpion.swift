//
//  Scorpion.swift
//  ScorpionKit
//
//  Façade:
//  - `inspect`: a model link, headers only — formats, adapter layout, family, fetch plan.
//  - `extractAdapter`: a model link's adapter weights as ΔW per module. Weight extraction is
//    format-level and model-agnostic (safetensors over HTTP ranges, kohya/PEFT/LyCORIS
//    conventions); executors map the result onto their own layers.
//  - `detectMemorization`: resolve a backend, encode the reference, run the memorization
//    probe (and optionally the seed basin), and return a `MemorizationReport`.
//

import CoreGraphics
import Foundation
import MLX

public enum ScorpionInfo {
    public static let version = "0.2.0"
}

public struct ScorpionOptions: Sendable {
    /// Maximum tensor bytes fetched when inspecting (decimal units, matching reported sizes).
    public var budgetBytes = 256_000_000
    public var auth = FetchAuth.fromEnvironment()
    public var cacheDirectory = FetchCache.defaultDirectory
    /// Keyed seed schedule; nil loads the per-install key (see `SeedSchedule.load`).
    public var seedSchedule: SeedSchedule?

    public init() {}
}

public enum ScanPhase: String, Sendable, CaseIterable {
    case resolving = "Resolving model link"
    case readingHeaders = "Reading tensor headers"
    case readingMetadata = "Reading model card"
    case fetchingTensors = "Fetching adapter tensors"
    case loadingModel = "Loading model"
    case probing = "Measuring memorization"
    case basin = "Seed basin"
    case done = "Done"
}

public struct ScanProgress: Sendable {
    public let phase: ScanPhase
    public let detail: String
    public let bytesFetched: Int
    public let weightBytes: Int?

    public init(phase: ScanPhase, detail: String, bytesFetched: Int = 0, weightBytes: Int? = nil) {
        self.phase = phase
        self.detail = detail
        self.bytesFetched = bytesFetched
        self.weightBytes = weightBytes
    }
}

public typealias ProgressHandler = @Sendable (ScanProgress) -> Void

public struct SourceSummary: Codable, Sendable {
    public let link: String
    public let host: ModelHost
    public let repo: String
    public let commit: String?
    public let files: [FileInventoryNote]
    public let repoBytes: Int
    public let weightBytes: Int
    public let bytesFetched: Int
    public let bytesFromCache: Int
    public let requests: Int

    /// Fraction of the model's weight bytes that crossed the network.
    public var fetchedFraction: Double? {
        weightBytes > 0 ? Double(bytesFetched) / Double(weightBytes) : nil
    }
}

public struct UnitPlan: Codable, Sendable {
    public let unit: String
    public let format: AdapterFormat
    public let moduleCount: Int
    public let familyKey: String
    public let conditioning: ConditioningSummary
    public let plannedModules: Int
    public let plannedBytes: Int
    public let unitBytes: Int
    public let sampleModules: [String]
    public let trainingMetadataKeys: [String]
}

public struct ModelInspection: Codable, Sendable {
    public let source: SourceSummary
    public let units: [UnitPlan]
    public let metadata: MetadataFindings
    public let flags: [String]
    public let ggufSummaries: [String: [String: String]]
}

/// An adapter's weights, decoded to one ΔW per module (names as in the file).
public struct ExtractedAdapter {
    public let source: SourceSummary
    public let unit: String
    public let format: AdapterFormat
    public let modules: [ModuleSpec]
    public let updates: [String: WeightUpdate]
    public let findings: MetadataFindings
    public let notes: [String]
}

public enum ScorpionError: Error, LocalizedError {
    case noAdapter(String)

    public var errorDescription: String? {
        switch self {
        case .noAdapter(let link): return "No adapter weights found at \(link)."
        }
    }
}

public final class Scorpion: @unchecked Sendable {
    public let options: ScorpionOptions
    public let schedule: SeedSchedule
    let fetcher: RangeFetcher

    public init(options: ScorpionOptions = .init()) {
        self.options = options
        self.schedule = options.seedSchedule ?? (try? SeedSchedule.load()) ?? SeedSchedule.ephemeral()
        self.fetcher = RangeFetcher(auth: options.auth, cache: FetchCache(directory: options.cacheDirectory))
    }

    deinit { fetcher.invalidate() }

    // MARK: - Model side

    struct LoadedModel {
        let source: ResolvedSource
        let inventory: TensorInventory
        let findings: MetadataFindings
        let peftScale: ((Int) -> Float)?
    }

    func loadModel(_ link: String, progress: ProgressHandler?) async throws -> LoadedModel {
        func report(_ phase: ScanPhase, _ detail: String, weightBytes: Int? = nil) {
            progress?(ScanProgress(phase: phase, detail: detail, bytesFetched: fetcher.ledger.networkBytes, weightBytes: weightBytes))
        }
        report(.resolving, link)
        let source = try await SourceResolver(fetcher: fetcher).resolve(link)
        report(.readingHeaders, "\(source.weightFiles.count) weight file(s), \(ByteFormat.string(source.weightBytes))",
               weightBytes: source.weightBytes)
        let inventory = try await InventoryReader(fetcher: fetcher).read(source)

        report(.readingMetadata, "model card and adapter config", weightBytes: source.weightBytes)
        var readme: String?
        if let file = source.file(named: "README.md") ?? source.file(named: "readme.md"),
           let data = try? await fetcher.fetchSmallFile(file.url, limit: 1 << 20, cacheable: source.pinned) {
            readme = String(decoding: data, as: UTF8.self)
        }
        var peftScale: ((Int) -> Float)?
        if let file = source.file(named: "adapter_config.json"),
           let data = try? await fetcher.fetchSmallFile(file.url, cacheable: source.pinned) {
            peftScale = AdapterDecoder.peftScale(fromConfig: data)
        }
        var metadata: [String: String] = [:]
        for unit in inventory.units { metadata.merge(unit.metadata) { a, _ in a } }
        let card = source.cardData.isEmpty ? readme.map(MetadataProbe.frontMatter) ?? [:] : source.cardData
        let findings = MetadataProbe.findings(safetensorsMetadata: metadata, cardData: card,
                                              readme: readme, hubTags: source.hubTags)
        return LoadedModel(source: source, inventory: inventory, findings: findings, peftScale: peftScale)
    }

    func sourceSummary(_ source: ResolvedSource, inventory: TensorInventory) -> SourceSummary {
        SourceSummary(link: source.reference.original, host: source.reference.host, repo: source.reference.repo,
                      commit: source.commit, files: inventory.notes, repoBytes: source.repoBytes,
                      weightBytes: source.weightBytes, bytesFetched: fetcher.ledger.networkBytes,
                      bytesFromCache: fetcher.ledger.cachedBytes, requests: fetcher.ledger.requests)
    }

    /// Header-only inspection: formats, layout, family, fetch plan. Fetches no weights.
    public func inspect(model link: String, progress: ProgressHandler? = nil) async throws -> ModelInspection {
        let loaded = try await loadModel(link, progress: progress)
        let analyzer = WeightAnalyzer(fetcher: fetcher, budgetBytes: options.budgetBytes)
        let plans = loaded.inventory.units.map { unit -> UnitPlan in
            let p = analyzer.plan(unit)
            return UnitPlan(unit: unit.name, format: p.layout.format, moduleCount: p.layout.modules.count,
                            familyKey: StructuralHash.familyKey(p.layout.modules),
                            conditioning: ConditioningSummary(method: p.heuristic.method, dims: p.heuristic.conditioningDims,
                                                              candidateCount: p.heuristic.candidates.count),
                            plannedModules: p.selected.count, plannedBytes: p.bytes, unitBytes: unit.byteSize,
                            sampleModules: Array(p.selected.prefix(6).map(\.name)),
                            trainingMetadataKeys: unit.metadata.keys.sorted())
        }
        var gguf: [String: [String: String]] = [:]
        for (path, header) in loaded.inventory.ggufHeaders {
            var s: [String: String] = ["tensors": String(header.tensors.count), "version": String(header.version)]
            for key in ["general.architecture", "general.name", "general.file_type"] {
                if let v = header.metadata[key] { s[key] = v.description }
            }
            gguf[path] = s
        }
        progress?(ScanProgress(phase: .done, detail: "", bytesFetched: fetcher.ledger.networkBytes, weightBytes: loaded.source.weightBytes))
        return ModelInspection(source: sourceSummary(loaded.source, inventory: loaded.inventory), units: plans,
                               metadata: loaded.findings, flags: loaded.inventory.flags, ggufSummaries: gguf)
    }

    /// The largest adapter unit at `link`, every module decoded to ΔW (all of its tensors are fetched).
    public func extractAdapter(model link: String, progress: ProgressHandler? = nil) async throws -> ExtractedAdapter {
        let loaded = try await loadModel(link, progress: progress)
        let withLayout = loaded.inventory.units.map { ($0, AdapterLayoutDetector.detect($0.tensors)) }
        guard let (unit, layout) = withLayout.filter({ $0.1.format != .fullModel && $0.1.format != .textualInversion })
            .max(by: { $0.0.byteSize < $1.0.byteSize }) else { throw ScorpionError.noAdapter(link) }
        let weightBytes = loaded.source.weightBytes
        fetcher.ledger.onProgress = { bytes in
            progress?(ScanProgress(phase: .fetchingTensors, detail: unit.name, bytesFetched: bytes, weightBytes: weightBytes))
        }
        defer { fetcher.ledger.onProgress = nil }
        let analyzer = WeightAnalyzer(fetcher: fetcher, budgetBytes: .max)
        let names = Set(layout.modules.flatMap(\.tensorNames))
        let tensors = try await analyzer.fetchTensors(names.compactMap { unit.tensors[$0] }, cacheable: loaded.source.pinned)
        let decoder = AdapterDecoder(peftScale: loaded.peftScale)
        var updates: [String: WeightUpdate] = [:]
        var notes: [String] = []
        for spec in layout.modules {
            do {
                updates[spec.name] = try decoder.decode(spec, tensors: tensors)
            } catch {
                notes.append("\(spec.name): \(error.localizedDescription)")
            }
        }
        if layout.modules.contains(where: \.hasDoRA) { notes.append("DoRA magnitudes present; direction-only approximation") }
        let ignored = Set(layout.modules.flatMap(\.ignoredFactors))
        if !ignored.isEmpty { notes.append("ignored factors: \(ignored.sorted().joined(separator: ", "))") }
        return ExtractedAdapter(source: sourceSummary(loaded.source, inventory: loaded.inventory), unit: unit.name,
                                format: layout.format, modules: layout.modules, updates: updates, findings: loaded.findings,
                                notes: notes)
    }

    // MARK: - Memorization

    /// Resolve `request.backend`, encode the reference (and controls) with the target's
    /// encoder, run the memorization probe, and — when `basin` is given and a region counts —
    /// the seed-basin test on the strongest region.
    public func detectMemorization(_ request: BackendRequest, controls: [ReferenceImage] = [],
                                   configuration: MemorizationConfiguration, basin: SeedBasinConfiguration? = nil,
                                   registry: BackendRegistry = .shared,
                                   progress: ProgressHandler? = nil) async throws -> MemorizationReport {
        progress?(ScanProgress(phase: .loadingModel, detail: request.backend))
        let models = try await registry.resolve(request)
        let reference = try models.target.encode(request.reference)
        let encodedControls = try controls.map { try models.target.encode($0) }
        let prompt = ProbePrompt.resolve(user: request.prompt, triggers: models.triggers)
        let result = try MemorizationProbe(configuration: configuration).run(
            models: models, reference: reference, prompt: prompt, controls: encodedControls,
            imageSize: CGSize(width: request.reference.width, height: request.reference.height), schedule: schedule,
            progress: { progress?(ScanProgress(phase: .probing, detail: $0)) })

        var evidence: SeedBasinEvidence?
        if let basin, result.regions.contains(where: \.counts) {
            progress?(ScanProgress(phase: .basin, detail: "inverting the reference and perturbing its seed"))
            var samples = encodedControls.map(\.latent)
            if samples.count < 2 {
                samples += try NullCalibration.baseSamples(models.base, like: reference, prompt: prompt, count: 8,
                                                           sampler: configuration.controlSampler, schedule: schedule).map(\.latent)
            }
            evidence = try SeedBasinTest(configuration: basin).run(models: models, reference: reference, result: result,
                                                                   scaleSamples: samples, schedule: schedule)
        }
        progress?(ScanProgress(phase: .done, detail: result.verdict.rawValue))
        return MemorizationReport(reference: ImageInfo(request.reference), controls: controls.map(ImageInfo.init),
                                  backend: request.backend, model: request.model, provenance: models.provenance,
                                  result: result, basin: evidence)
    }
}
