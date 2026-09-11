//
//  Scorpion.swift
//  ScorpionKit
//
//  Façade: inspect a model link (headers only), analyze a reference image, and scan the
//  pair into a LikenessReport. Phase 1 is model-agnostic: nothing below knows any
//  architecture — only file formats, adapter conventions, linear algebra and text.
//

import Foundation
import MLX

public enum ScorpionInfo {
    public static let version = "0.1.0"
}

public struct ScanOptions: Sendable {
    /// Maximum tensor bytes fetched per scan (decimal units, matching reported sizes).
    public var budgetBytes = 256_000_000
    public var permutations = 64
    public var band = LogSNRBand.identity
    /// Face to analyze when the reference has several (default: largest).
    public var faceIndex: Int?
    public var useCLIP = true
    public var clipRepo = CLIPModel.defaultRepo
    public var bankDirectory: URL? = FingerprintBank.defaultDirectory
    public var calibration: Calibration?
    public var maxUnits = 4
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
    case fetchingTensors = "Fetching selected tensors"
    case loadingCLIP = "Loading CLIP"
    case analyzingReference = "Analyzing reference"
    case matching = "Matching"
    case scoring = "Scoring"
    case done = "Done"
}

public struct ScanProgress: Sendable {
    public let phase: ScanPhase
    public let detail: String
    public let bytesFetched: Int
    public let weightBytes: Int?
}

public typealias ProgressHandler = @Sendable (ScanProgress) -> Void

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

public struct ReferenceAnalysis: Codable, Sendable {
    public let referenceID: String
    public let name: String
    public let width: Int
    public let height: Int
    public let faces: [FaceRegion]
    public let selectedFace: Int?
    public let descriptors: [Descriptor]
    public let visualStats: VisualStats
    public let crops: [CropSpec]
    public let seedPlan: SeedPlan
    public let prompts: PromptSet
    public let segmentation: [FaceSegmentSummary]
    public let clipModel: String?
    public let notes: [String]
}

public final class Scorpion: @unchecked Sendable {
    public let options: ScanOptions
    public let schedule: SeedSchedule
    let fetcher: RangeFetcher
    private var describer: Describer?
    private let lock = NSLock()

    public init(options: ScanOptions = .init()) {
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

    /// Units worth analyzing: adapters first (largest first), else the largest checkpoint.
    func selectUnits(_ inventory: TensorInventory) -> [WeightUnit] {
        let withLayout = inventory.units.map { ($0, AdapterLayoutDetector.detect($0.tensors).format) }
        let adapters = withLayout.filter { $0.1 != .fullModel }.map(\.0).sorted { $0.byteSize > $1.byteSize }
        if !adapters.isEmpty { return Array(adapters.prefix(options.maxUnits)) }
        return Array(inventory.units.sorted { $0.byteSize > $1.byteSize }.prefix(1))
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

    /// Full models diff against the base named in their card, when its tensors line up.
    func findBase(for unit: WeightUnit, findings: MetadataFindings) async -> WeightUnit? {
        guard let base = findings.baseModel, base.split(separator: "/").count == 2, !base.contains(" "),
              let ref = try? SourceParser.parse("https://huggingface.co/\(base)"),
              let source = try? await SourceResolver(fetcher: fetcher).resolve(ref),
              let inventory = try? await InventoryReader(fetcher: fetcher).read(source) else { return nil }
        let wanted = unit.tensors.values.filter { $0.shape.count >= 2 }
        let best = inventory.units.max { a, b in overlap(a, wanted) < overlap(b, wanted) }
        guard let best, Double(overlap(best, wanted)) >= 0.8 * Double(max(wanted.count, 1)) else { return nil }
        return best
    }

    private func overlap(_ unit: WeightUnit, _ wanted: [TensorRecord]) -> Int {
        wanted.filter { unit.tensors[$0.name]?.shape == $0.shape }.count
    }

    // MARK: - Reference side

    /// Loaded describers are shared process-wide (keyed by CLIP repo): loading weights and
    /// embedding the descriptor vocabulary is the expensive part of reference analysis.
    private static let describerLock = NSLock()
    nonisolated(unsafe) private static var describers: [String: Describer] = [:]

    public func loadDescriber(progress: ProgressHandler? = nil) async throws -> Describer {
        if let d = lock.withLock({ describer }) { return d }
        if let shared = Self.describerLock.withLock({ Self.describers[options.clipRepo] }) {
            lock.withLock { describer = shared }
            return shared
        }
        progress?(ScanProgress(phase: .loadingCLIP, detail: options.clipRepo, bytesFetched: fetcher.ledger.networkBytes, weightBytes: nil))
        let clip = try await CLIPModel.load(repo: options.clipRepo, cacheDirectory: options.cacheDirectory) { fraction in
            progress?(ScanProgress(phase: .loadingCLIP, detail: String(format: "%@ %.0f%%", self.options.clipRepo, fraction * 100),
                                   bytesFetched: self.fetcher.ledger.networkBytes, weightBytes: nil))
        }
        let d = Describer(clip: clip)
        Self.describerLock.withLock { Self.describers[options.clipRepo] = d }
        lock.withLock { describer = d }
        return d
    }

    /// `triggers`: the model's trigger words, when known (scans pass them; `describe` doesn't).
    public func analyzeReference(_ reference: ReferenceImage, triggers: [String] = [],
                                 progress: ProgressHandler? = nil) async throws -> ReferenceAnalysis {
        var notes: [String] = []
        let faces = (try? FaceDetector.detect(reference)) ?? []
        let selected = faces.isEmpty ? nil : min(options.faceIndex ?? 0, faces.count - 1)
        let focus = selected.map { faces[$0] }

        var descriptors: [Descriptor] = []
        var clipName: String?
        if options.useCLIP {
            do {
                let describer = try await loadDescriber(progress: progress)
                progress?(ScanProgress(phase: .analyzingReference, detail: "describing via CLIP",
                                       bytesFetched: fetcher.ledger.networkBytes, weightBytes: nil))
                descriptors = describer.describe(reference, faces: focus.map { [$0] } ?? [])
                clipName = describer.clip.repo
            } catch {
                notes.append("CLIP unavailable (\(error.localizedDescription)); description skipped")
            }
        } else {
            notes.append("CLIP disabled; description skipped")
        }
        let stats = VisualStats.compute(reference)
        let crops = CropPlanner.crops(for: reference, faces: focus.map { [$0] } ?? [])
        let plan = SeedEngine.plan(schedule: schedule, descriptors: descriptors, count: options.permutations,
                                   band: options.band, cropCount: crops.count, triggers: triggers)
        if faces.isEmpty { notes.append("no face detected; region analysis uses the image center") }
        let segmentation = focus.map { [FaceSegmenter.segment(reference, face: $0).summary] } ?? []
        return ReferenceAnalysis(referenceID: reference.id, name: reference.name, width: reference.width,
                                 height: reference.height, faces: faces, selectedFace: selected,
                                 descriptors: descriptors, visualStats: stats, crops: crops, seedPlan: plan,
                                 prompts: PromptSet.make(descriptors: descriptors, triggers: triggers),
                                 segmentation: segmentation, clipModel: clipName, notes: notes)
    }

    // MARK: - Scan

    public func scan(model link: String, image: URL, progress: ProgressHandler? = nil) async throws -> LikenessReport {
        try await scan(model: link, reference: ReferenceImage.load(url: image), progress: progress)
    }

    public func scan(model link: String, reference: ReferenceImage, progress: ProgressHandler? = nil) async throws -> LikenessReport {
        let loaded = try await loadModel(link, progress: progress)
        let weightBytes = loaded.source.weightBytes
        fetcher.ledger.onProgress = { bytes in
            progress?(ScanProgress(phase: .fetchingTensors, detail: "", bytesFetched: bytes, weightBytes: weightBytes))
        }
        defer { fetcher.ledger.onProgress = nil }

        // One budget for the whole scan, shared across units in plan order.
        var analyses: [WeightAnalysis] = []
        var remaining = options.budgetBytes
        for unit in selectUnits(loaded.inventory) where remaining > 0 {
            progress?(ScanProgress(phase: .fetchingTensors, detail: unit.name, bytesFetched: fetcher.ledger.networkBytes, weightBytes: weightBytes))
            var base: WeightUnit?
            if AdapterLayoutDetector.detect(unit.tensors).format == .fullModel {
                base = await findBase(for: unit, findings: loaded.findings)
            }
            let analyzer = WeightAnalyzer(fetcher: fetcher, budgetBytes: remaining)
            let analysis = try await analyzer.analyze(unit, base: base, cacheable: loaded.source.pinned,
                                                      peftScale: loaded.peftScale)
            remaining -= analysis.summary.bytesPlanned
            analyses.append(analysis)
        }

        let ref = try await analyzeReference(reference, triggers: loaded.findings.triggerWords, progress: progress)
        progress?(ScanProgress(phase: .matching, detail: "metadata and fingerprint bank", bytesFetched: fetcher.ledger.networkBytes, weightBytes: weightBytes))
        let match = MetadataMatcher.match(loaded.findings, descriptors: ref.descriptors,
                                          clip: lock.withLock { describer?.clip })

        var neighbors: [BankNeighbor] = []
        if let dir = options.bankDirectory, let bank = try? FingerprintBank(directory: dir) {
            for a in analyses where !a.fingerprints.isEmpty {
                neighbors += bank.topK(a.fingerprints, familyKey: a.summary.familyKey)
            }
            neighbors.sort { $0.similarity > $1.similarity }
        }

        progress?(ScanProgress(phase: .scoring, detail: "", bytesFetched: fetcher.ledger.networkBytes, weightBytes: weightBytes))
        let components = Self.components(findings: loaded.findings, match: match, analyses: analyses.map(\.summary),
                                         neighbors: neighbors, reference: ref, calibration: options.calibration)
        let harm = ScoreFusion.fuse(components, calibration: options.calibration)
        let report = LikenessReport(
            schema: LikenessReport.schema, reportID: UUID(), timestamp: Date(), scorpionVersion: ScorpionInfo.version,
            reference: ReferenceSummary(referenceID: ref.referenceID, name: ref.name, width: ref.width, height: ref.height,
                                        faces: ref.faces, selectedFace: ref.selectedFace, descriptors: ref.descriptors,
                                        visualStats: ref.visualStats, seedSchedule: ref.seedPlan.schedule, prompts: ref.prompts,
                                        segmentation: ref.segmentation,
                                        permutations: ref.seedPlan.permutations.count, band: ref.seedPlan.band,
                                        clipModel: ref.clipModel),
            source: sourceSummary(loaded.source, inventory: loaded.inventory),
            units: analyses.map(\.summary), flags: loaded.inventory.flags + (loaded.findings.nsfwIndicators.isEmpty ? [] : ["nsfw-tags"]),
            metadata: loaded.findings, metadataMatch: match, bankNeighbors: Array(neighbors.prefix(5)),
            probes: ProbeSection(likelihood: nil,
                                 note: "No denoiser executor for this model family yet: the likelihood ratio (Δbits), the "
                                     + "face-seed needle and memory pools run on analytic models via `scorpion research gmm` / "
                                     + "`research needle` / `research memory`; real executors plug into DiffusionBackend in Phase 2."),
            components: components, harmScore: harm, harmScoreCI90: nil,
            calibrated: (options.calibration?.fittedOn ?? 0) > 0, disclaimer: LikenessReport.disclaimer)
        progress?(ScanProgress(phase: .done, detail: "", bytesFetched: fetcher.ledger.networkBytes, weightBytes: weightBytes))
        return report
    }

    static func components(findings: MetadataFindings, match: MetadataMatchResult?, analyses: [UnitAnalysis],
                           neighbors: [BankNeighbor], reference: ReferenceAnalysis,
                           calibration: Calibration?) -> [ScoreComponent] {
        let weights = (calibration ?? .provisional).weights
        func w(_ k: ScoreComponent.Kind) -> Double { weights[k.rawValue] ?? 0 }
        func clamp(_ v: Double) -> Double { min(max(v, 0.02), 0.98) }
        var out: [ScoreComponent] = []

        if let match {
            let top = match.matches.prefix(3).map { "\($0.descriptor)↔\($0.term)" }.joined(separator: ", ")
            out.append(.init(kind: .metadataMatch, value: clamp(match.score), weight: w(.metadataMatch),
                             note: top.isEmpty ? "no descriptor matched the model's metadata" : "matches: \(top)"))
        }
        if !findings.sources.isEmpty || !findings.hubTags.isEmpty {
            let person = !findings.personIndicators.isEmpty
            let refHasPerson = !reference.faces.isEmpty || reference.descriptors.contains {
                $0.category == "subject" && DescriptorVocabulary.bundled.personPhrases.contains($0.phrase)
            }
            let value = person ? (refHasPerson ? 0.85 : 0.6) : 0.25
            out.append(.init(kind: .personAdapter, value: value, weight: w(.personAdapter),
                             note: person ? "person indicators: \(findings.personIndicators.prefix(5).joined(separator: ", "))"
                                          : "no person/likeness indicators in metadata"))
        }
        if let best = analyses.compactMap({ a in a.aggregate.map { (a, $0) } }).max(by: { $0.1.narrowness < $1.1.narrowness }) {
            out.append(.init(kind: .spectralNarrowness, value: clamp(best.1.narrowness), weight: w(.spectralNarrowness),
                             note: String(format: "%@: %d conditioning modules, mean effective rank %.1f, concentration %.2f",
                                          best.0.unit, best.1.modules, best.1.meanEffectiveRank, best.1.meanConcentration)))
        }
        if let vote = FingerprintBank.likenessVote(neighbors) {
            out.append(.init(kind: .bankVote, value: clamp(vote), weight: w(.bankVote),
                             note: "nearest: " + neighbors.prefix(3).map { String(format: "%@ (%.2f)", $0.label, $0.similarity) }
                                .joined(separator: ", ")))
        }
        return out
    }

    // MARK: - Bank

    /// Fingerprint a model's adapter units into the bank under `label`.
    public func addToBank(model link: String, label: String, progress: ProgressHandler? = nil) async throws -> [FingerprintEntry] {
        guard let dir = options.bankDirectory else { return [] }
        let loaded = try await loadModel(link, progress: progress)
        let bank = try FingerprintBank(directory: dir)
        let analyzer = WeightAnalyzer(fetcher: fetcher, budgetBytes: options.budgetBytes)
        var added: [FingerprintEntry] = []
        for unit in selectUnits(loaded.inventory) {
            let a = try await analyzer.analyze(unit, cacheable: loaded.source.pinned, peftScale: loaded.peftScale)
            guard !a.fingerprints.isEmpty else { continue }
            let source = "\(loaded.source.reference.repo)@\(loaded.source.commit ?? "unpinned")"
            added.append(try bank.add(label: label, familyKey: a.summary.familyKey, source: source,
                                      unit: unit.name, fingerprints: a.fingerprints))
        }
        return added
    }
}
