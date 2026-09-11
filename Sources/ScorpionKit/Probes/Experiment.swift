//
//  Experiment.swift
//  ScorpionKit
//
//  `scorpion experiment run`: the deciding experiment from Docs/EXPERIMENT.md as code.
//  A manifest lists (model, base, reference set, controls, label) pairs; every pair gets
//  the needle probe (Δbits, ι, footprint, CI, cost) and face-region Δbits; the runner
//  reports AUCs (identity-match vs look-alike, vs unrelated) and a go/no-go verdict against
//  the protocol's four criteria.
//
//  Real model families plug in through `BackendRegistry` (Phase 2 executors). Until then a
//  `toy` block synthesizes a world — several identities, their look-alikes, unrelated
//  people, one "LoRA" per identity and a style-only model — so the entire analysis
//  pipeline runs end to end and the verdict logic can be checked on known answers.
//

import Foundation
import MLX

public struct ExperimentManifest: Codable, Sendable {
    public enum Label: String, Codable, Sendable {
        case identityMatch = "identity-match"
        case lookAlike = "look-alike"
        case unrelated
        /// A photo the fine-tune trained on (memory pools, Tier M).
        case member
        /// A photo of the same person the fine-tune never saw (Tier M).
        case heldOut = "held-out"
    }

    /// Tier M arm: identity fine-tunes at several memorization strengths κ (the toy's stand-in
    /// for training checkpoints — memorization sets in with training time, τ_mem ∝ n).
    public struct Memorization: Codable, Sendable {
        public var kappas: [Double] = [0, 0.5, 1]
        public var members = 8
        public var duplicates = 1
        public var identityWeight = 0.05
        /// Member and held-out references per identity and κ.
        public var references = 4
        /// Hutchinson probes per level.
        public var probes = 4

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kappas = try c.decodeIfPresent([Double].self, forKey: .kappas) ?? kappas
            members = try c.decodeIfPresent(Int.self, forKey: .members) ?? members
            duplicates = try c.decodeIfPresent(Int.self, forKey: .duplicates) ?? duplicates
            identityWeight = try c.decodeIfPresent(Double.self, forKey: .identityWeight) ?? identityWeight
            references = try c.decodeIfPresent(Int.self, forKey: .references) ?? references
            probes = try c.decodeIfPresent(Int.self, forKey: .probes) ?? probes
        }
    }

    public struct Toy: Codable, Sendable {
        public var identities = 3
        public var photosPerIdentity = 5
        public var lookAlikesPerIdentity = 2
        public var lookAlikeDistance: Float = 1.0
        public var unrelated = 3
        public var needleWeight = 1e-2
        public var variance: Float = 0.05
        public var side = 12
        public var generic = 12
        public var styleModel = true
        /// Run the needle / likeness arm (default true).
        public var needleArm: Bool? = nil
        /// Run the memory-pool arm when present.
        public var memorization: Memorization? = nil
    }

    public struct Pair: Codable, Sendable {
        public var id: String
        public var backend: String
        public var base: String
        public var references: [String]
        public var controls: [String]
        public var label: Label
    }

    public struct Settings: Codable, Sendable {
        public var samplesPerLevel = 100
        public var maxLevels = 4
        public var steps = 10
        public var guidance: [Float] = [1]
        public var faceBand = LogSNRBand(lo: -6, hi: 8)
        public var faceProbePermutations = 64
    }

    public var name: String
    public var toy: Toy?
    public var settings: Settings?
    public var pairs: [Pair]?
}

public struct ExperimentReport: Codable, Sendable {
    public struct Row: Codable, Sendable {
        public let pair: String
        public let label: ExperimentManifest.Label
        public let needleDeltaBits: Double
        public let needleIsLowerBound: Bool
        public let faceDeltaBits: Double
        public let backgroundIndependence: Double?
        public let footprintFraction: Double
        public let ciHalfWidthBits: Double
        public let earlyExitAgreement: Double?
        public let denoiserCalls: Int
        public let seconds: Double
    }

    public struct Criterion: Codable, Sendable {
        public let name: String
        public let requirement: String
        public let value: String
        public let pass: Bool
    }

    /// One reference scored by the memory-pool probe (Tier M).
    public struct MemoryRow: Codable, Sendable {
        public let pair: String
        public let label: ExperimentManifest.Label
        public let kappa: Double
        /// Sustained face collapse (pool signal).
        public let collapse: Double
        /// Band snap (single-photo membership statistic).
        public let snap: Double
        public let iou: Double
        public let tier: MemoryPoolResult.Tier
        public let denoiserCalls: Int
    }

    public let name: String
    public let rows: [Row]
    public let auc: [String: Double]
    public let criteria: [Criterion]
    public let go: Bool
    public let notes: [String]
    public let seedSchedule: ScheduleInfo
    public var memoryRows: [MemoryRow]? = nil
    /// Tier M verdict, separate from the needle's GO (neither replaces the other).
    public var memoryCriteria: [Criterion]? = nil
    public var memoryGo: Bool? = nil
}

public enum ExperimentError: Error, LocalizedError {
    case noExecutor(String)
    case empty

    public var errorDescription: String? {
        switch self {
        case .noExecutor(let name):
            return "No executor registered for backend '\(name)'. Real model executors arrive in Phase 2; use a `toy` block for now."
        case .empty: return "The manifest has neither a toy block nor pairs."
        }
    }
}

/// Resolves backend names ("prefix:details") to executors. Phase 2 registers real families.
public final class BackendRegistry: @unchecked Sendable {
    public typealias Factory = (String) throws -> DiffusionBackend
    private var factories: [String: Factory] = [:]
    private let lock = NSLock()

    public static let shared = BackendRegistry()

    public init() {}

    public func register(prefix: String, factory: @escaping Factory) {
        lock.withLock { factories[prefix] = factory }
    }

    public func resolve(_ name: String) throws -> DiffusionBackend {
        let prefix = String(name.split(separator: ":").first ?? "")
        guard let factory = lock.withLock({ factories[prefix] }) else { throw ExperimentError.noExecutor(name) }
        return try factory(name)
    }
}

/// Several identities in one analytic world, for dry-running the experiment.
struct ToyWorld {
    enum Kind { case identity, lookAlike, unrelated }

    struct Person {
        let id: String
        let kind: Kind
        let of: Int?        // identity index this person resembles (look-alikes)
        let mean: [Float]
    }

    let spec: ExperimentManifest.Toy
    let grid: [Float]
    let people: [Person]
    let crowd: [[Float]]
    let base: AnalyticGMMBackend
    let loras: [AnalyticGMMBackend]
    let style: AnalyticGMMBackend?

    var shape: [Int] { [1, spec.side, spec.side] }

    init(spec: ExperimentManifest.Toy, seed: UInt64) {
        var rng = SplitMix64(seed: seed)
        let side = spec.side, d = side * side
        let lo = side / 4, hi = side - side / 4
        let grid: [Float] = (0..<d).map { i in (lo..<hi).contains(i / side) && (lo..<hi).contains(i % side) ? 1 : 0 }
        let crowd = (0..<spec.generic).map { _ in GMMScenario.proceduralField(side: side, rng: &rng) }
        var people: [Person] = []
        for i in 0..<spec.identities {
            let mean = GMMScenario.proceduralField(side: side, rng: &rng)
            people.append(Person(id: "identity-\(i)", kind: .identity, of: i, mean: mean))
            for j in 0..<spec.lookAlikesPerIdentity {
                let bg = crowd[(i + j) % crowd.count]
                let alike = (0..<d).map { k in
                    grid[k] > 0.5 ? mean[k] + spec.lookAlikeDistance * GMMScenario.gaussian(&rng) : bg[k]
                }
                people.append(Person(id: "identity-\(i)-lookalike-\(j)", kind: .lookAlike, of: i, mean: alike))
            }
        }
        for u in 0..<spec.unrelated {
            people.append(Person(id: "unrelated-\(u)", kind: .unrelated, of: nil,
                                 mean: GMMScenario.proceduralField(side: side, rng: &rng)))
        }

        // The base knows the crowd, the look-alikes and unrelated people — not the identities.
        let known = crowd + people.filter { $0.kind != .identity }.map(\.mean)
        let build = { (id: String, extra: [Float]?, background: [[Float]]) in
            Self.model(id, spec: spec, grid: grid, known: known, extra: extra, background: background)
        }
        self.spec = spec
        self.grid = grid
        self.crowd = crowd
        self.people = people
        self.base = build("toy-base", nil, crowd)
        self.loras = people.filter { $0.kind == .identity }.map { build("toy-lora-\($0.id)", $0.mean, crowd) }
        self.style = spec.styleModel ? build("toy-style", nil, crowd.map { $0.map { $0 + 0.8 } }) : nil
    }

    static func model(_ id: String, spec: ExperimentManifest.Toy, grid: [Float], known: [[Float]],
                      extra: [Float]?, background: [[Float]]) -> AnalyticGMMBackend {
        let pi = extra == nil ? 0 : spec.needleWeight
        let w = [Double](repeating: (1 - pi) / Double(known.count), count: known.count) + (extra == nil ? [] : [pi])
        return AnalyticGMMBackend(identifier: id, side: spec.side, blocks: [
            .init(label: "face", indicator: grid, means: known + (extra.map { [$0] } ?? []), variance: spec.variance, weights: w),
            .init(label: "rest", indicator: grid.map { 1 - $0 }, means: background, variance: spec.variance),
        ])
    }

    func photos(_ p: Person, count: Int, schedule: SeedSchedule, index: Int) -> [[Float]] {
        var rng = schedule.rng(.jitter, index)
        let sd = spec.variance.squareRoot()
        return (0..<count).map { _ in p.mean.map { $0 + sd * GMMScenario.gaussian(&rng) } }
    }

    /// The people most similar to `p` (face coordinates), for control sets.
    func nearest(to p: Person, count: Int) -> [Person] {
        func dist(_ a: [Float], _ b: [Float]) -> Float {
            zip(zip(a, b), grid).reduce(0) { $0 + ($1.1 > 0.5 ? ($1.0.0 - $1.0.1) * ($1.0.0 - $1.0.1) : 0) }
        }
        return people.filter { $0.id != p.id }.sorted { dist($0.mean, p.mean) < dist($1.mean, p.mean) }.prefix(count).map { $0 }
    }
}

public enum ExperimentRunner {
    public static func run(_ manifest: ExperimentManifest, registry: BackendRegistry = .shared,
                           schedule: SeedSchedule = .research,
                           progress: ((String) -> Void)? = nil) throws -> ExperimentReport {
        guard let toy = manifest.toy else {
            if let first = manifest.pairs?.first { _ = try registry.resolve(first.backend) }
            throw ExperimentError.empty
        }
        var report = (toy.needleArm ?? true)
            ? try runNeedleArm(manifest, toy: toy, schedule: schedule, progress: progress)
            : verdict(name: manifest.name, rows: [], schedule: schedule)
        if let memorization = toy.memorization {
            let rows = try runMemoryArm(toy: toy, memorization: memorization, schedule: schedule, progress: progress)
            let (criteria, go) = memoryVerdict(rows)
            report.memoryRows = rows
            report.memoryCriteria = criteria
            report.memoryGo = go
        }
        return report
    }

    /// Tier M on the toy: identities fine-tuned at each κ; their training photos (members)
    /// and unseen photos (held-out) go through the memory-pool probe.
    static func runMemoryArm(toy: ExperimentManifest.Toy, memorization m: ExperimentManifest.Memorization,
                             schedule: SeedSchedule, progress: ((String) -> Void)?) throws -> [ExperimentReport.MemoryRow] {
        let side = toy.side, d = side * side
        let lo = side / 4, hi = side - side / 4
        let grid: [Float] = (0..<d).map { i in (lo..<hi).contains(i / side) && (lo..<hi).contains(i % side) ? 1 : 0 }
        var rng = SplitMix64(seed: schedule.value(.scenario, 2))
        var config = MemoryPoolConfiguration()
        config.probes = m.probes
        config.logSNRs = [2, 3, 4, 5, 6, 7, 8]
        config.checkBasin = false
        config.capture = false
        let probe = MemoryPoolProbe(configuration: config)
        let prompts = PromptSet(descriptor: "a photo", triggers: [])
        var rows: [ExperimentReport.MemoryRow] = []
        for i in 0..<toy.identities {
            let face = GMMScenario.proceduralField(side: side, rng: &rng)
            for kappa in m.kappas {
                progress?(String(format: "[memory] identity %d, κ = %.2f", i, kappa))
                let s = MemoryScenario.build(referencePixels: face, referenceIsMember: true, faceGrid: grid, side: side,
                                             generic: toy.generic, lookAlikes: toy.lookAlikesPerIdentity,
                                             lookAlikeDistance: toy.lookAlikeDistance, members: m.members, heldOut: m.references,
                                             kappa: kappa, duplicates: m.duplicates, identityWeight: m.identityWeight,
                                             variance: toy.variance, seed: schedule.value(.scenario, 100 + i))
                let sets: [(ExperimentManifest.Label, [[Float]])] = [(.member, Array(s.members.prefix(m.references))),
                                                                      (.heldOut, Array(s.heldOut.prefix(m.references)))]
                for (label, refs) in sets {
                    for (j, ref) in refs.enumerated() {
                        let r = try probe.run(target: s.target, base: s.base, reference: MLXArray(ref, s.latentShape),
                                              face: s.faceMask, footprint: nil, prompts: prompts, schedule: schedule)
                        rows.append(.init(pair: String(format: "toy-memory-%d-k%.2f × %@-%d", i, kappa, label.rawValue, j),
                                          label: label, kappa: kappa, collapse: r.peakSignal, snap: r.bandSnap, iou: r.iou,
                                          tier: r.tier, denoiserCalls: r.evaluations))
                    }
                }
            }
        }
        return rows
    }

    /// Tier M go/no-go (Docs/EXPERIMENT.md): membership, localization, dose-response, specificity.
    static func memoryVerdict(_ rows: [ExperimentReport.MemoryRow]) -> ([ExperimentReport.Criterion], Bool) {
        func median(_ v: [Double]) -> Double? {
            guard !v.isEmpty else { return nil }
            let s = v.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        let kappas = Array(Set(rows.map(\.kappa))).sorted()
        let strongest = rows.filter { $0.kappa == kappas.last }
        let members = strongest.filter { $0.label == .member }, held = strongest.filter { $0.label == .heldOut }
        let auc = ScoreFusion.auc(scores: (members + held).map(\.snap),
                                  labels: members.map { _ in true } + held.map { _ in false }) ?? 0
        let pooled = members.filter { $0.collapse >= 0.1 }
        let iou = median(pooled.map(\.iou)) ?? 0
        let medians = kappas.map { k in median(rows.filter { $0.kappa == k && $0.label == .member }.map(\.collapse)) ?? 0 }
        let rho = spearman(kappas, medians)
        let pools: Set<MemoryPoolResult.Tier> = [.memorizedPhoto, .identityPool, .nearbyPool]
        let learned = rows.filter { $0.kappa == kappas.first }
        let falsePools = learned.isEmpty ? 1 : Double(learned.filter { pools.contains($0.tier) }.count) / Double(learned.count)
        let criteria: [ExperimentReport.Criterion] = [
            .init(name: "membership", requirement: "AUC(member vs held-out) ≥ 0.90 on band snap, strongest memorization",
                  value: String(format: "%.3f", auc), pass: auc >= 0.9),
            .init(name: "localization", requirement: "median IoU(collapse map, face) ≥ 0.50 for members in a pool",
                  value: String(format: "%.2f (%d pools)", iou, pooled.count), pass: iou >= 0.5),
            .init(name: "dose-response", requirement: "Spearman(memorization, median member collapse) ≥ 0.80",
                  value: String(format: "%.2f", rho), pass: kappas.count >= 2 && rho >= 0.8),
            .init(name: "specificity", requirement: "≤ 10% of photos flagged as pools at the least memorization",
                  value: String(format: "%.0f%%", falsePools * 100), pass: falsePools <= 0.1),
        ]
        return (criteria, criteria.allSatisfy(\.pass))
    }

    static func spearman(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count >= 2 else { return 0 }
        func ranks(_ v: [Double]) -> [Double] {
            let order = v.indices.sorted { v[$0] < v[$1] }
            var r = [Double](repeating: 0, count: v.count)
            for (rank, i) in order.enumerated() { r[i] = Double(rank) }
            return r
        }
        let rx = ranks(x), ry = ranks(y)
        let mx = rx.reduce(0, +) / Double(rx.count), my = ry.reduce(0, +) / Double(ry.count)
        let cov = zip(rx, ry).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
        let vx = rx.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }, vy = ry.reduce(0) { $0 + ($1 - my) * ($1 - my) }
        return vx > 0 && vy > 0 ? cov / (vx * vy).squareRoot() : 0
    }

    static func runNeedleArm(_ manifest: ExperimentManifest, toy: ExperimentManifest.Toy, schedule: SeedSchedule,
                             progress: ((String) -> Void)?) throws -> ExperimentReport {
        let settings = manifest.settings ?? .init()
        let world = ToyWorld(spec: toy, seed: schedule.value(.scenario, 1))
        var config = NeedleConfiguration()
        config.subset.samplesPerLevel = settings.samplesPerLevel
        config.subset.maxLevels = settings.maxLevels
        config.sampler = DeterministicSampler(solver: .dpmSolver2M, steps: settings.steps)
        config.guidance = settings.guidance
        config.checkReachability = false

        struct Job {
            let id: String
            let model: AnalyticGMMBackend
            let person: ToyWorld.Person
            let label: ExperimentManifest.Label
        }
        var jobs: [Job] = []
        let identities = world.people.filter { $0.kind == .identity }
        for (i, identity) in identities.enumerated() {
            let lora = world.loras[i]
            jobs.append(Job(id: "\(lora.identifier) × \(identity.id)", model: lora, person: identity, label: .identityMatch))
            for alike in world.people where alike.kind == .lookAlike && alike.of == i {
                jobs.append(Job(id: "\(lora.identifier) × \(alike.id)", model: lora, person: alike, label: .lookAlike))
            }
            for other in world.people where other.kind == .unrelated {
                jobs.append(Job(id: "\(lora.identifier) × \(other.id)", model: lora, person: other, label: .unrelated))
            }
            if let style = world.style {
                jobs.append(Job(id: "\(style.identifier) × \(identity.id)", model: style, person: identity, label: .unrelated))
            }
        }

        let mask = MLXArray(world.grid, world.shape)
        let footprint = SeedFootprint.fixed("face", weights: world.grid, shape: world.shape)
        var rows: [ExperimentReport.Row] = []
        for (n, job) in jobs.enumerated() {
            let started = Date()
            progress?("[\(n + 1)/\(jobs.count)] \(job.id)")
            let pIndex = world.people.firstIndex { $0.id == job.person.id }!
            let refs = world.photos(job.person, count: toy.photosPerIdentity, schedule: schedule, index: 1_000 + pIndex)
            let refArray = MLXArray(refs.flatMap { $0 }, [refs.count] + world.shape)
            var criterion = LatentFidelity(references: refArray, mask: mask, threshold: 0, aggregate: .mean)
            let positives = world.photos(job.person, count: 30, schedule: schedule, index: 2_000 + pIndex)
            let negatives = world.nearest(to: job.person, count: 3).enumerated().flatMap {
                world.photos($1, count: 10, schedule: schedule, index: 3_000 + 10 * pIndex + $0)
            }
            let cal = LatentFidelity.calibrate(
                positive: criterion.scores(MLXArray(positives.flatMap { $0 }, [positives.count] + world.shape)),
                negative: criterion.scores(MLXArray(negatives.flatMap { $0 }, [negatives.count] + world.shape)))
            criterion.threshold = cal.threshold

            let x0 = refArray[0]
            let needle = try NeedleProbe(configuration: config).run(
                target: job.model, base: world.base, reference: x0, criterion: criterion, thresholdMethod: cal.method,
                calibration: (cal.tpr, cal.fpr), footprint: footprint,
                prompts: PromptSet(descriptor: "a photo", triggers: []), schedule: schedule)
            let plan = SeedEngine.plan(schedule: schedule, descriptors: [], count: settings.faceProbePermutations,
                                       band: settings.faceBand)
            let face = try BandLikelihoodProbe().run(target: job.model, base: world.base, latent: x0,
                                                     masks: [("face", mask)], plan: plan)
            let t = needle.target[0].estimate
            let calls = needle.evaluations * settings.steps * ((settings.guidance.first ?? 1) == 1 ? 1 : 2)
                + 2 * settings.faceProbePermutations
            rows.append(.init(pair: job.id, label: job.label, needleDeltaBits: needle.deltaBits[0].value,
                              needleIsLowerBound: needle.deltaBits[0].isLowerBound,
                              faceDeltaBits: face.region("face")?.deltaBits ?? 0,
                              backgroundIndependence: needle.target[0].backgroundIndependence,
                              footprintFraction: footprint.fraction,
                              ciHalfWidthBits: (t.log2CI90[1] - t.log2CI90[0]) / 2,
                              earlyExitAgreement: needle.earlyExitAgreement, denoiserCalls: calls,
                              seconds: Date().timeIntervalSince(started)))
        }
        return verdict(name: manifest.name, rows: rows, schedule: schedule)
    }

    static func verdict(name: String, rows: [ExperimentReport.Row], schedule: SeedSchedule) -> ExperimentReport {
        guard !rows.isEmpty else {
            return ExperimentReport(name: name, rows: [], auc: [:], criteria: [], go: false,
                                    notes: ["Needle / likeness arm not run (toy.needleArm = false)."], seedSchedule: schedule.info)
        }
        func auc(_ value: (ExperimentReport.Row) -> Double, against negative: ExperimentManifest.Label) -> Double? {
            let pos = rows.filter { $0.label == .identityMatch }, neg = rows.filter { $0.label == negative }
            return ScoreFusion.auc(scores: (pos + neg).map(value), labels: pos.map { _ in true } + neg.map { _ in false })
        }
        var aucs: [String: Double] = [:]
        aucs["needle: match vs look-alike"] = auc(\.needleDeltaBits, against: .lookAlike)
        aucs["needle: match vs unrelated"] = auc(\.needleDeltaBits, against: .unrelated)
        aucs["face: match vs look-alike"] = auc(\.faceDeltaBits, against: .lookAlike)
        aucs["face: match vs unrelated"] = auc(\.faceDeltaBits, against: .unrelated)

        func median(_ v: [Double]) -> Double? {
            guard !v.isEmpty else { return nil }
            let s = v.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        let matches = rows.filter { $0.label == .identityMatch }
        let discrimination = max(aucs["needle: match vs look-alike"] ?? 0, aucs["face: match vs look-alike"] ?? 0)
        let footprint = median(matches.map(\.footprintFraction)) ?? 1
        let independence = median(matches.compactMap(\.backgroundIndependence)) ?? 0
        let halfWidth = median(matches.map(\.ciHalfWidthBits)) ?? .infinity
        let maxCalls = rows.map(\.denoiserCalls).max() ?? 0
        let early = rows.compactMap(\.earlyExitAgreement).min() ?? 0

        let criteria: [ExperimentReport.Criterion] = [
            .init(name: "discrimination", requirement: "AUC(identity-match vs look-alike) ≥ 0.90 (needle or face Δbits)",
                  value: String(format: "%.3f", discrimination), pass: discrimination >= 0.9),
            .init(name: "locality", requirement: "median |F|/d ≤ 0.35 and median ι ≥ 0.70",
                  value: String(format: "|F|/d %.2f, ι %.2f", footprint, independence),
                  pass: footprint <= 0.35 && independence >= 0.7),
            .init(name: "precision & cost", requirement: "median bits CI half-width ≤ 1.5 within ≤ 20k denoiser calls per pair",
                  value: String(format: "±%.2f bits, max %d calls", halfWidth, maxCalls),
                  pass: halfWidth <= 1.5 && maxCalls <= 20_000),
            .init(name: "early exit", requirement: "early-exit agreement ≥ 0.95",
                  value: String(format: "%.3f", early), pass: early >= 0.95),
        ]
        return ExperimentReport(name: name, rows: rows, auc: aucs, criteria: criteria, go: criteria.allSatisfy(\.pass),
                                notes: ["Toy dry run: validates the analysis pipeline and verdict logic, not real models.",
                                        "Lower-bound Δbits (base never reached the needle) enter the AUCs at their bound."],
                                seedSchedule: schedule.info)
    }
}
