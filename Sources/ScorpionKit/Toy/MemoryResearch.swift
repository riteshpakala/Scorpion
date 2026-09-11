//
//  MemoryResearch.swift
//  ScorpionKit
//
//  `scorpion research memory`: the memorization probe on analytic worlds built around the
//  user's own image, where memorization is a known dial. For each strength κ it plants a
//  memorized region and builds two worlds — the image was a training member, or it was held
//  out (the fine-tune saw only other photos of the subject) — and runs the full probe on
//  both, plus a light audit over other members, held-out photos, look-alikes and strangers
//  for the membership AUC. Side by side with the truth: the exact collapse and its map.
//

import CoreGraphics
import Foundation
import MLX

public struct MemoryResearchOptions: Codable, Sendable {
    public var side = 16
    public var generic = 16
    public var lookAlikes = 4
    public var lookAlikeDistance: Float = 1.0
    public var variance: Float = 0.05
    public var kappas: [Double] = [0, 0.5, 0.9, 1]
    public var members = 12
    public var duplicates = 1
    /// Planted region in normalized frame coordinates (x, y, width, height).
    public var plant: [Double] = [0.25, 0.25, 0.5, 0.5]
    public var identityWeight = 0.05
    /// Trigger-gated fine-tune (conditional overfitting); nil = every prompt sees the memory.
    public var trigger: String? = "sks person"
    /// Smoothed-subject weight the unconditional model keeps when gated.
    public var leak = 1e-3
    /// References per kind in the membership audit.
    public var audit = 6
    public var configuration = MemorizationConfiguration.toy

    public init() {}

    /// The cheaper probe the audit runs (Hutchinson K = 4, the levels pools live at).
    var auditConfiguration: MemorizationConfiguration {
        var c = configuration
        c.probes = 4
        c.scoutLevels = [2, 3, 4, 5, 6, 7, 8]
        return c
    }
}

public struct MemoryResearchReport: Codable, Sendable {
    public struct Audit: Codable, Sendable {
        public let kind: String
        public let maxMass: Double
        public let membershipSnap: Double
        public let verdict: MemorizationVerdict
    }

    public struct Row: Codable, Sendable {
        public let kappa: Double
        /// Pool width h² of each memorized photo.
        public let bandwidth: Float
        public let member: MemorizationResult
        public let heldOut: MemorizationResult
        /// AUC(training members vs held-out photos of the same subject) on membership snap.
        public let membershipAUC: Double?
        /// Share of each audit kind flagged (memorized or memorized-other-content).
        public let flagRate: [String: Double]
        /// IoU of the member probe's counted regions with the planted region.
        public let iou: Double
        /// Exact collapse over the planted region at the member's strongest level, and the
        /// estimated peak collapse of its strongest region.
        public let exactPeakCollapse: Double
        public let estimatedPeakCollapse: Double?
        /// IoU of the exact collapse map's top cells (as many as planted) with the planted region.
        public let exactIoU: Double
        public let audit: [Audit]
    }

    public let referenceID: String
    public let side: Int
    public let plantedCells: Int
    public let options: MemoryResearchOptions
    public let rows: [Row]
    public let seedSchedule: ScheduleInfo
}

public enum MemoryResearch {
    public static func run(reference: ReferenceImage, options: MemoryResearchOptions = .init(),
                           schedule: SeedSchedule = .research, progress: ((String) -> Void)? = nil) throws -> MemoryResearchReport {
        let side = options.side
        let refPixels = AnalyticGMMBackend.encodePixels(reference, rect: reference.bounds, side: side)
        let p = options.plant
        let prompt = ProbePrompt.resolve(user: options.trigger, triggers: [])
        let frame = reference.bounds

        func world(kappa: Double, member: Bool) -> MemoryScenario {
            let plant = PlantedRegion.rect("plant", x: p[0], y: p[1], width: p[2], height: p[3], side: side, kappa: kappa)
            return MemoryScenario.build(referencePixels: refPixels, referenceIsMember: member, regions: [plant], side: side,
                                        generic: options.generic, lookAlikes: options.lookAlikes,
                                        lookAlikeDistance: options.lookAlikeDistance, members: options.members,
                                        heldOut: options.audit, duplicates: options.duplicates,
                                        identityWeight: options.identityWeight, variance: options.variance,
                                        trigger: options.trigger, leak: options.leak, seed: schedule.value(.scenario, 10))
        }
        func probe(_ s: MemoryScenario, _ pixels: [Float], _ config: MemorizationConfiguration) throws -> MemorizationResult {
            let ref = EncodedReference(latent: MLXArray(pixels, s.latentShape), frame: frame)
            return try MemorizationProbe(configuration: config).run(
                models: ModelPair(target: s.target, base: s.base), reference: ref, prompt: prompt,
                imageSize: CGSize(width: reference.width, height: reference.height), schedule: schedule)
        }

        var rows: [MemoryResearchReport.Row] = []
        var plantedCells = 0
        for kappa in options.kappas {
            progress?(String(format: "κ = %.2f: reference as a training member", kappa))
            let m = world(kappa: kappa, member: true)
            plantedCells = m.regions[0].cellCount
            let member = try probe(m, refPixels, options.configuration)
            progress?(String(format: "κ = %.2f: reference held out", kappa))
            let heldOut = try probe(world(kappa: kappa, member: false), refPixels, options.configuration)

            progress?(String(format: "κ = %.2f: membership audit", kappa))
            let n = options.audit
            let alikes = m.lookAlikeMeans.prefix(n).enumerated().flatMap {
                m.photos(of: $1, count: 1, variance: m.variance, schedule: schedule, index: 700 + $0)
            }
            let strangers = m.crowdMeans.prefix(n).enumerated().flatMap {
                m.photos(of: $1, count: 1, variance: m.variance, schedule: schedule, index: 800 + $0)
            }
            let sets: [(String, [[Float]])] = [("member", Array(m.members.dropFirst().prefix(n))),
                                               ("held-out", Array(m.heldOut.prefix(n))),
                                               ("look-alike", alikes), ("stranger", strangers)]
            var audit: [MemoryResearchReport.Audit] = []
            for (kind, refs) in sets {
                for r in refs {
                    let result = try probe(m, r, options.auditConfiguration)
                    audit.append(.init(kind: kind, maxMass: result.regions.map(\.mass).max() ?? 0,
                                       membershipSnap: result.membershipSnap, verdict: result.verdict))
                }
            }
            let members = audit.filter { $0.kind == "member" }, held = audit.filter { $0.kind == "held-out" }
            let auc = Statistics.auc(scores: (members + held).map(\.membershipSnap),
                                     labels: members.map { _ in true } + held.map { _ in false })

            // Truth for the member probe at its strongest level.
            let planted = m.regions[0].grid.map { $0 > 0.5 }
            let lambda = member.regions.first?.peakLogSNR ?? member.levels.max { $0.meanCollapse < $1.meanCollapse }?.logSNR ?? 4
            let draws = options.configuration.draws
            let (xt, _, _) = CollapseFieldStage.noised(m.target, x0: MLXArray(refPixels, m.latentShape), logSNR: lambda,
                                                       draws: draws, schedule: schedule)
            let l = MLXArray([Float](repeating: lambda, count: draws))
            let c = [Conditioning(prompt: prompt.text)]
            let exact = (m.base.exactRetainedVariance(xt, logSNR: l, conditions: c)
                - m.target.exactRetainedVariance(xt, logSNR: l, conditions: c)).mean(axis: 0).reshaped(-1).asArray(Float.self)
            let exactPlanted = zip(exact, planted).filter { $0.1 }.map { Double($0.0) }
            let counted = Set(member.regions.filter(\.counts).flatMap(\.cells))
            rows.append(.init(
                kappa: kappa, bandwidth: m.bandwidths[0], member: member, heldOut: heldOut, membershipAUC: auc,
                flagRate: Dictionary(grouping: audit, by: \.kind).mapValues { group in
                    Double(group.filter { $0.verdict == .memorized || $0.verdict == .memorizedOtherContent }.count) / Double(group.count)
                },
                iou: GridMath.iou(planted.indices.map { counted.contains($0) }, planted),
                exactPeakCollapse: exactPlanted.reduce(0, +) / Double(max(exactPlanted.count, 1)),
                estimatedPeakCollapse: member.regions.first?.meanCollapse,
                exactIoU: topIoU(exact, planted), audit: audit))
        }
        return MemoryResearchReport(referenceID: reference.id, side: side, plantedCells: plantedCells, options: options,
                                    rows: rows, seedSchedule: schedule.info)
    }

    /// IoU of the top-|truth| positions of `map` with `truth`.
    static func topIoU(_ map: [Float], _ truth: [Bool]) -> Double {
        let k = truth.filter { $0 }.count
        guard k > 0, k < truth.count else { return 0 }
        let top = Set(map.indices.sorted { map[$0] > map[$1] }.prefix(k))
        let inter = top.filter { truth[$0] }.count
        return Double(inter) / Double(2 * k - inter)
    }
}
