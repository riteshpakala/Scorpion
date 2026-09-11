//
//  Experiment.swift
//  ScorpionKit
//
//  `scorpion experiment run`: the go/no-go protocol from Docs/EXPERIMENT.md as code. Subjects
//  are fine-tuned at several memorization strengths (the toy's stand-in for training
//  checkpoints — memorization sets in with training time, τ_mem ∝ n); their training photos
//  (members) and unseen photos (held-out) go through the memorization probe, and the runner
//  checks the four criteria: membership, localization, dose-response, specificity.
//
//  Until the real-model arm is wired to executors, a `toy` block synthesizes the worlds so
//  the whole analysis runs end to end and the verdict logic is checked on known answers.
//

import CoreGraphics
import Foundation
import MLX

public struct ExperimentManifest: Codable, Sendable {
    public enum Label: String, Codable, Sendable {
        /// A photo the fine-tune trained on.
        case member
        /// A photo of the same subject the fine-tune never saw.
        case heldOut = "held-out"
    }

    public struct Memorization: Codable, Sendable {
        public var kappas: [Double] = [0, 0.5, 1]
        public var members = 8
        public var duplicates = 1
        public var identityWeight = 0.05
        /// Member and held-out references per subject and κ.
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
        public var lookAlikesPerIdentity = 2
        public var lookAlikeDistance: Float = 1.0
        public var variance: Float = 0.05
        public var side = 12
        public var generic = 12
        public var memorization = Memorization()

        public init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            identities = try c.decodeIfPresent(Int.self, forKey: .identities) ?? identities
            lookAlikesPerIdentity = try c.decodeIfPresent(Int.self, forKey: .lookAlikesPerIdentity) ?? lookAlikesPerIdentity
            lookAlikeDistance = try c.decodeIfPresent(Float.self, forKey: .lookAlikeDistance) ?? lookAlikeDistance
            variance = try c.decodeIfPresent(Float.self, forKey: .variance) ?? variance
            side = try c.decodeIfPresent(Int.self, forKey: .side) ?? side
            generic = try c.decodeIfPresent(Int.self, forKey: .generic) ?? generic
            memorization = try c.decodeIfPresent(Memorization.self, forKey: .memorization) ?? memorization
        }
    }

    public var name: String
    public var toy: Toy?
}

public struct ExperimentReport: Codable, Sendable {
    public struct Row: Codable, Sendable {
        public let pair: String
        public let label: ExperimentManifest.Label
        public let kappa: Double
        /// Mean collapse of the strongest counted region (0 when none).
        public let collapse: Double
        /// Membership snap (confirmation draws).
        public let snap: Double
        /// IoU of the counted regions with the planted region.
        public let iou: Double
        public let verdict: MemorizationVerdict
        public let denoiserCalls: Int
    }

    public struct Criterion: Codable, Sendable {
        public let name: String
        public let requirement: String
        public let value: String
        public let pass: Bool
    }

    public let name: String
    public let rows: [Row]
    public let criteria: [Criterion]
    public let go: Bool
    public let notes: [String]
    public let seedSchedule: ScheduleInfo
}

public enum ExperimentError: Error, LocalizedError {
    case empty

    public var errorDescription: String? {
        "The manifest has no toy block. Real-model pairs run through `scorpion memorization --backend …` until the experiment runner is wired to executors."
    }
}

public enum ExperimentRunner {
    public static func run(_ manifest: ExperimentManifest, schedule: SeedSchedule = .research,
                           progress: ((String) -> Void)? = nil) throws -> ExperimentReport {
        guard let toy = manifest.toy else { throw ExperimentError.empty }
        let m = toy.memorization
        let side = toy.side
        var rng = SplitMix64(seed: schedule.value(.scenario, 2))
        var config = MemorizationConfiguration.toy
        config.probes = m.probes
        config.scoutLevels = [2, 3, 4, 5, 6, 7, 8]
        let probe = MemorizationProbe(configuration: config)
        let prompt = ProbePrompt(text: "a photo", source: "user")
        let frame = CGRect(x: 0, y: 0, width: side, height: side)
        var rows: [ExperimentReport.Row] = []
        for i in 0..<toy.identities {
            let subject = ToyField.proceduralField(side: side, rng: &rng)
            for kappa in m.kappas {
                progress?(String(format: "subject %d, κ = %.2f", i, kappa))
                let plant = PlantedRegion.center(side: side, kappa: kappa)
                let s = MemoryScenario.build(referencePixels: subject, referenceIsMember: true, regions: [plant], side: side,
                                             generic: toy.generic, lookAlikes: toy.lookAlikesPerIdentity,
                                             lookAlikeDistance: toy.lookAlikeDistance, members: m.members, heldOut: m.references,
                                             duplicates: m.duplicates, identityWeight: m.identityWeight,
                                             variance: toy.variance, seed: schedule.value(.scenario, 100 + i))
                let planted = s.regions[0].grid.map { $0 > 0.5 }
                let sets: [(ExperimentManifest.Label, [[Float]])] = [(.member, Array(s.members.prefix(m.references))),
                                                                      (.heldOut, Array(s.heldOut.prefix(m.references)))]
                for (label, refs) in sets {
                    for (j, ref) in refs.enumerated() {
                        let r = try probe.run(models: ModelPair(target: s.target, base: s.base),
                                              reference: EncodedReference(latent: MLXArray(ref, s.latentShape), frame: frame),
                                              prompt: prompt, schedule: schedule)
                        let counted = r.regions.filter(\.counts)
                        let cells = Set(counted.flatMap(\.cells))
                        rows.append(.init(pair: String(format: "toy-memory-%d-k%.2f × %@-%d", i, kappa, label.rawValue, j),
                                          label: label, kappa: kappa, collapse: counted.first?.meanCollapse ?? 0,
                                          snap: r.membershipSnap,
                                          iou: GridMath.iou(planted.indices.map { cells.contains($0) }, planted),
                                          verdict: r.verdict, denoiserCalls: r.evaluations))
                    }
                }
            }
        }
        let (criteria, go) = verdict(rows)
        return ExperimentReport(name: manifest.name, rows: rows, criteria: criteria, go: go,
                                notes: ["Toy dry run: validates the analysis pipeline and verdict logic, not real models."],
                                seedSchedule: schedule.info)
    }

    /// Go/no-go (Docs/EXPERIMENT.md): membership, localization, dose-response, specificity.
    static func verdict(_ rows: [ExperimentReport.Row]) -> ([ExperimentReport.Criterion], Bool) {
        let kappas = Array(Set(rows.map(\.kappa))).sorted()
        let strongest = rows.filter { $0.kappa == kappas.last }
        let members = strongest.filter { $0.label == .member }, held = strongest.filter { $0.label == .heldOut }
        let auc = Statistics.auc(scores: (members + held).map(\.snap), labels: members.map { _ in true } + held.map { _ in false }) ?? 0
        let pooled = rows.filter { $0.label == .member && $0.collapse > 0 }
        let iou = Statistics.median(pooled.map(\.iou)) ?? 0
        let medians = kappas.map { k in Statistics.median(rows.filter { $0.kappa == k && $0.label == .member }.map(\.collapse)) ?? 0 }
        let rho = Statistics.spearman(kappas, medians)
        let least = rows.filter { $0.kappa == kappas.first }
        let flagged: Set<MemorizationVerdict> = [.memorized, .memorizedOtherContent]
        let falseFlags = least.isEmpty ? 1 : Double(least.filter { flagged.contains($0.verdict) }.count) / Double(least.count)
        let criteria: [ExperimentReport.Criterion] = [
            .init(name: "membership", requirement: "AUC(member vs held-out) ≥ 0.90 on membership snap, strongest memorization",
                  value: String(format: "%.3f", auc), pass: auc >= 0.9),
            .init(name: "localization", requirement: "median IoU(counted regions, planted region) ≥ 0.50 for members with a region",
                  value: String(format: "%.2f (%d regions)", iou, pooled.count), pass: iou >= 0.5),
            .init(name: "dose-response", requirement: "Spearman(memorization, median member collapse) ≥ 0.80",
                  value: String(format: "%.2f", rho), pass: kappas.count >= 2 && rho >= 0.8),
            .init(name: "specificity", requirement: "≤ 10% of photos flagged at the least memorization",
                  value: String(format: "%.0f%%", falseFlags * 100), pass: falseFlags <= 0.1),
        ]
        return (criteria, criteria.allSatisfy(\.pass))
    }
}
