//
//  MemoryResearch.swift
//  ScorpionKit
//
//  `scorpion research memory`: memory pools on analytic models built around the user's own
//  photo — real face segmentation, real extra photos when given — where memorization is a
//  known dial. For each memorization strength κ it builds two worlds:
//  - the photo was a training member (the fine-tune saw it, plus other photos of the face);
//  - the photo was held out (the fine-tune saw only other photos of the face);
//  and runs the full probe on both, plus a light audit over other members, held-out photos,
//  look-alikes and strangers for membership and pool AUCs. Side by side with the truth: the
//  exact collapse and its map, the Hutchinson error, and the exact face log-likelihood ratio.
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
    public var scope = MemoryScenario.Scope.face
    /// Identity weight under the trigger (or under every prompt when ungated).
    public var identityWeight = 0.05
    /// Trigger-gated fine-tune (conditional overfitting); nil = every prompt sees the memory.
    public var trigger: String? = "sks person"
    /// Smoothed-identity weight the unconditional model keeps when gated.
    public var leak = 1e-3
    /// Reference-set size for the identity criterion; missing photos are synthetic.
    public var photos = 5
    /// References per kind in the membership / pool audit.
    public var audit = 6
    public var memory = MemoryPoolConfiguration()

    public init() {
        memory.needle.sampler = DeterministicSampler(solver: .dpmSolver2M, steps: 32)
        memory.needle.subset.samplesPerLevel = 150
        memory.needle.subset.maxLevels = 5
        memory.basinTrials = 16
        memory.basinSampler = DeterministicSampler(solver: .ddim, steps: 96)
        memory.captureKinds = ["trigger", "empty"]
    }
}

public struct MemoryResearchReport: Codable, Sendable {
    public struct Audit: Codable, Sendable {
        public let kind: String
        public let signal: Double
        public let snap: Double
    }

    public struct Row: Codable, Sendable {
        public let kappa: Double
        /// Pool width h² of each memorized photo.
        public let bandwidth: Float
        public let member: MemoryPoolResult
        public let heldOut: MemoryPoolResult
        /// AUC(training members vs held-out photos of the same person) on band snap.
        public let membershipAUC: Double?
        /// AUC(photos of the person vs look-alikes and strangers) on the pool signal (rank only).
        public let poolAUC: Double?
        /// Share of each audit kind flagged as sitting in a pool (signal ≥ threshold).
        public let poolRate: [String: Double]
        /// Exact face collapse at the member probe's peak, and the estimate's relative error.
        public let exactPeakCollapse: Double
        public let collapseError: Double?
        public let exactIoU: Double
        /// Face Δbits (band likelihood estimate) and the exact log-likelihood ratio, member photo.
        public let faceDeltaBits: Double
        public let exactBits: Double
        public let audit: [Audit]
    }

    public let referenceID: String
    public let segmentation: FaceSegmentSummary?
    public let regionLabel: String
    public let faceFraction: Double
    public let latentShape: [Int]
    public let options: MemoryResearchOptions
    public let rows: [Row]
    public let referencePhotos: Int
    public let seedSchedule: ScheduleInfo
}

public enum MemoryResearch {
    public static func run(reference: ReferenceImage, extraPhotos: [ReferenceImage] = [],
                           options: MemoryResearchOptions = .init(), schedule: SeedSchedule = .research,
                           progress: ((String) -> Void)? = nil) throws -> MemoryResearchReport {
        let side = options.side
        let faces = (try? FaceDetector.detect(reference)) ?? []
        let segment = faces.first.map { FaceSegmenter.segment(reference, face: $0) }
        let region = segment?.face ?? RegionMask.center(of: reference)
        let crop = faces.first.flatMap { CropPlanner.crops(for: reference, faces: [$0]).dropFirst().first } ?? reference.centerSquare
        let refPixels = AnalyticGMMBackend.encodePixels(reference, rect: crop.rect, side: side)
        let grid = region.grid(crop: crop.rect, width: side, height: side)
        func encodeFace(_ image: ReferenceImage) -> [Float] {
            let f = (try? FaceDetector.detect(image)) ?? []
            let rect = f.first.flatMap { CropPlanner.crops(for: image, faces: [$0]).dropFirst().first?.rect } ?? image.centerSquare.rect
            return AnalyticGMMBackend.encodePixels(image, rect: rect, side: side)
        }
        let extras = extraPhotos.map(encodeFace)
        let prompts = PromptSet(descriptor: "a photo of a person", triggers: options.trigger.map { [$0] } ?? [])

        func world(kappa: Double, member: Bool) -> MemoryScenario {
            MemoryScenario.build(referencePixels: refPixels, referenceIsMember: member, faceGrid: grid, side: side,
                                 generic: options.generic, lookAlikes: options.lookAlikes,
                                 lookAlikeDistance: options.lookAlikeDistance, members: options.members,
                                 heldOut: options.audit, kappa: kappa, duplicates: options.duplicates,
                                 identityWeight: options.identityWeight, variance: options.variance, scope: options.scope,
                                 trigger: options.trigger, leak: options.leak, seed: schedule.value(.scenario, 10))
        }
        func features(_ s: MemoryScenario) -> [SeedFootprint] {
            (segment?.features ?? [:]).sorted { $0.key.rawValue < $1.key.rawValue }.map { feature, mask in
                SeedFootprint.fixed(feature.rawValue, weights: mask.grid(crop: crop.rect, width: side, height: side),
                                    shape: s.latentShape)
            }
        }
        /// Identity criterion (reference set vs controls) and the photo-level scale.
        func criteria(_ s: MemoryScenario, reference: [Float]) -> (LatentFidelity, Double) {
            var set = [reference] + extras
            set += s.photos(of: s.identityMean, count: max(0, options.photos - set.count), schedule: schedule, index: 2)
            let refs = MLXArray(set.flatMap { $0 }, [set.count] + s.latentShape)
            var identity = LatentFidelity(references: refs, mask: s.faceMask, threshold: 0, aggregate: .mean)
            let pos = s.photos(of: s.identityMean, count: 40, schedule: schedule, index: 1)
            let neg = s.controlPhotos(schedule: schedule)
            identity.threshold = LatentFidelity.calibrate(
                positive: identity.scores(MLXArray(pos.flatMap { $0 }, [pos.count] + s.latentShape)),
                negative: identity.scores(MLXArray(neg.flatMap { $0 }, [neg.count] + s.latentShape))).threshold
            return (identity, MemoryPoolProbe.photoScale(refs, mask: s.faceMask) ?? 2 * Double(s.spread))
        }
        func probe(_ s: MemoryScenario, reference: [Float], full: Bool) throws -> MemoryPoolResult {
            let x0 = MLXArray(reference, s.latentShape)
            let (identity, scale) = criteria(s, reference: reference)
            var config = options.memory
            config.needle.duplicateRadius = 0.1 * scale
            if !full {
                // Audit: a cheaper Hutchinson pass over the levels pools live at. (Forward-only
                // spread is confounded for faces new to the base — see RESEARCH.md §8.)
                config.probes = 4
                config.logSNRs = [2, 3, 4, 5, 6, 7, 8]
                config.checkBasin = false
                config.capture = false
            }
            return try MemoryPoolProbe(configuration: config).run(
                target: s.target, base: s.base, reference: x0, face: s.faceMask, features: features(s),
                identity: full ? identity : nil,
                photo: MemoryPoolProbe.photoCriterion(reference: x0, mask: s.faceMask, scale: scale),
                footprint: s.scope == .face ? .fixed(region.label, weights: s.faceGrid, shape: s.latentShape) : nil,
                prompts: prompts, schedule: schedule)
        }

        var rows: [MemoryResearchReport.Row] = []
        var faceFraction = 0.0
        for kappa in options.kappas {
            progress?(String(format: "κ = %.2f: reference as a training member", kappa))
            let m = world(kappa: kappa, member: true)
            faceFraction = m.faceFraction
            let member = try probe(m, reference: refPixels, full: true)
            progress?(String(format: "κ = %.2f: reference held out", kappa))
            let h = world(kappa: kappa, member: false)
            let heldOut = try probe(h, reference: refPixels, full: true)

            progress?(String(format: "κ = %.2f: membership and pool audit", kappa))
            var audit: [MemoryResearchReport.Audit] = []
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
            for (kind, refs) in sets {
                for r in refs {
                    let result = try probe(m, reference: r, full: false)
                    audit.append(.init(kind: kind, signal: result.peakSignal, snap: result.bandSnap))
                }
            }
            func auc(_ value: (MemoryResearchReport.Audit) -> Double, _ pos: Set<String>, _ neg: Set<String>) -> Double? {
                let p = audit.filter { pos.contains($0.kind) }, q = audit.filter { neg.contains($0.kind) }
                return ScoreFusion.auc(scores: (p + q).map(value), labels: p.map { _ in true } + q.map { _ in false })
            }

            // Truth for the member probe: exact collapse and map at its peak, exact likelihood ratio.
            let index = member.points.firstIndex { $0.logSNR == member.peakLogSNR } ?? 0
            let x0 = MLXArray(refPixels, m.latentShape)
            let draws = options.memory.noiseDraws
            let (xt, _, _) = MemoryPoolProbe.noised(m.target, x0: x0, logSNR: member.peakLogSNR, index: index,
                                                    draws: draws, schedule: schedule)
            let l = MLXArray([Float](repeating: member.peakLogSNR, count: draws))
            let c = [Conditioning(prompt: prompts.attacker)]
            let exact = m.base.exactRetainedVariance(xt, logSNR: l, conditions: c) - m.target.exactRetainedVariance(xt, logSNR: l, conditions: c)
            let face = m.faceMask.expandedDimensions(axis: 0)
            let exactFace = Double(((exact * face).sum() / (face.sum() * Float(draws))).item(Float.self))
            let exactMap = exact.mean(axis: 0).reshaped(-1).asArray(Float.self)
            let estimate = member.points[index].collapse?.face ?? member.points[index].spreadCollapse.face
            // Both under the attacker prompt: a gated memory only exists where the trigger opens it.
            let plan = SeedEngine.plan(schedule: schedule, descriptors: [], count: 64, band: LogSNRBand(lo: -6, hi: 8))
                .prompted(prompts.attacker)
            let bits = try BandLikelihoodProbe().run(target: m.target, base: m.base, latent: x0,
                                                     masks: [(region.label, m.faceMask)], plan: plan)

            rows.append(.init(
                kappa: kappa, bandwidth: m.bandwidth, member: member, heldOut: heldOut,
                membershipAUC: auc(\.snap, ["member"], ["held-out"]),
                poolAUC: auc(\.signal, ["member", "held-out"], ["look-alike", "stranger"]),
                poolRate: Dictionary(grouping: audit, by: \.kind).mapValues { group in
                    Double(group.filter { $0.signal >= options.memory.collapseThreshold }.count) / Double(group.count)
                },
                exactPeakCollapse: exactFace,
                collapseError: abs(exactFace) > 0.05 ? abs(estimate - exactFace) / abs(exactFace) : nil,
                exactIoU: MemoryPoolProbe.topIoU(exactMap, m.faceGrid.map { $0 > 0.5 }),
                faceDeltaBits: bits.region(region.label)?.deltaBits ?? 0,
                exactBits: (m.target.exactLogDensity(x0, conditions: c) - m.base.exactLogDensity(x0, conditions: c)) / log(2.0),
                audit: audit))
        }
        return MemoryResearchReport(referenceID: reference.id, segmentation: segment?.summary, regionLabel: region.label,
                                    faceFraction: faceFraction, latentShape: [1, side, side], options: options, rows: rows,
                                    referencePhotos: 1 + extraPhotos.count, seedSchedule: schedule.info)
    }
}
