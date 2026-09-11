//
//  NeedleProbe.swift
//  ScorpionKit
//
//  The face-seed needle. Split every seed into the segment that decides the face (the
//  measured footprint F) and the surrounding seed. The needle is the set of face segments
//  that produce the reference's face for (almost) any surrounding seed; its mass
//      μ = P_{z∼N(0,I)}( face of sample(z) matches the reference )
//  is estimated by subset simulation with paired random numbers for target and base:
//  - −log₂ μ: the entropy an attacker's seed must supply ("bits of luck needed");
//  - Δ = log₂ μ_T − log₂ μ_B: bits the target's weights save them;
//  - 1 − (1 − μ)^N: chance of success within N attempts (the worst-case view);
//  - ι: how often a needle face segment survives a fresh surrounding seed.
//  Reachability is checked separately: the inverted reference seed reproduces the face
//  under *both* models — every image is reachable; μ is what differs.
//
//  Identity test: latent fidelity only (no face is ever decoded). Fidelity to the reference
//  photo set, thresholded so look-alike controls don't count, approximates "this person";
//  its limit is quantified against the toy's identity oracle.
//

import Foundation
import MLX

public protocol NeedleCriterion {
    var name: String { get }
    var threshold: Double { get }
    /// Continuous scores for sampled latents (B, …latent); a hit is score ≥ threshold.
    func scores(_ x0: MLXArray) -> [Double]
}

/// Negative masked mean squared latent distance to the reference photo set.
/// `.nearest` scores closeness to any one photo (photo-level). `.mean` averages over the
/// set, which approximates distance to the identity's center plus its spread — several
/// photos of the same person define the identity better than one.
public struct LatentFidelity: NeedleCriterion {
    public enum Aggregate: String, Codable, Sendable {
        case nearest
        case mean
    }

    public var name: String { "latent-fidelity-\(aggregate.rawValue)" }
    /// (R, …latent)
    public let references: MLXArray
    /// (…latent) face weights.
    public let mask: MLXArray
    public var threshold: Double
    public var aggregate: Aggregate

    public init(references: MLXArray, mask: MLXArray, threshold: Double, aggregate: Aggregate = .mean) {
        self.references = references
        self.mask = mask
        self.threshold = threshold
        self.aggregate = aggregate
    }

    public func scores(_ x0: MLXArray) -> [Double] {
        let b = x0.dim(0), r = references.dim(0)
        let m = mask.expandedDimensions(axes: [0, 1])
        let diff = (x0.expandedDimensions(axis: 1) - references.expandedDimensions(axis: 0)).square() * m
        let d = diff.reshaped(b, r, -1).sum(axis: 2) / mask.sum()
        let agg = aggregate == .mean ? d.mean(axis: 1) : d.min(axis: 1)
        return (-agg).asArray(Float.self).map(Double.init)
    }

    /// Threshold for a *rare* event. A needle of mass π is swamped by look-alikes unless
    /// FPR × (look-alike prevalence) ≪ π, so a balanced (Youden) cut is the wrong rule. When
    /// controls and positives separate, the threshold sits midway between the closest control
    /// and the 10th percentile of positives (TPR ≥ 0.9, zero control hits). When they overlap,
    /// it falls back to the loosest cut with ≤ 1% control hits and says so.
    public static func calibrate(positive: [Double], negative: [Double])
        -> (threshold: Double, tpr: Double, fpr: Double, method: String) {
        guard !positive.isEmpty else { return (0, 0, 0, "none") }
        let pos = positive.sorted()
        let p10 = pos[Int(0.1 * Double(pos.count - 1))]
        func rates(_ t: Double) -> (Double, Double) {
            (Double(positive.filter { $0 >= t }.count) / Double(positive.count),
             negative.isEmpty ? 0 : Double(negative.filter { $0 >= t }.count) / Double(negative.count))
        }
        guard let maxNeg = negative.max() else {
            let (tpr, fpr) = rates(p10)
            return (p10, tpr, fpr, "positives-only")
        }
        if maxNeg < p10 {
            let t = (maxNeg + p10) / 2
            let (tpr, fpr) = rates(t)
            return (t, tpr, fpr, "separated")
        }
        let negDesc = negative.sorted(by: >)
        let allowed = Int(0.01 * Double(negative.count))
        let t = negDesc[min(allowed, negDesc.count - 1)] + 1e-9
        let (tpr, fpr) = rates(t)
        return (t, tpr, fpr, "overlap-fpr1%")
    }
}

/// Ground-truth identity (toy only): margin by which the identity component is nearest.
public struct OracleCriterion: NeedleCriterion {
    public let name = "identity-oracle"
    public let threshold = 0.0
    let margin: (MLXArray) -> [Float]

    public init(margin: @escaping (MLXArray) -> [Float]) { self.margin = margin }
    public func scores(_ x0: MLXArray) -> [Double] { margin(x0).map(Double.init) }
}

public struct NeedleConfiguration: Codable, Sendable {
    public var subset = SubsetSimulation()
    public var sampler = DeterministicSampler()
    public var guidance: [Float] = [1]
    /// Also decide hits at this λ (x̂₀ early exit) and report agreement with full sampling.
    public var earlyExitLogSNR: Float? = 2
    public var backgroundChecks = 4
    public var maxCheckedHits = 32
    /// Inversion needs fine steps: a needle's basin in seed space is small.
    public var reachabilitySteps = 128
    public var checkReachability = true
    public var attackAttempts = [1, 10, 100, 1_000, 10_000]
    /// When set, report how often hits recur as near-duplicates: another hit from a clearly
    /// different seed lands within this masked squared latent distance (memorized content
    /// comes back as copies; a learned identity varies).
    public var duplicateRadius: Double? = nil
    /// Minimum masked squared distance between two hits' face seeds for them to count as
    /// different seeds (independent seeds sit at ≈ 2).
    public var duplicateSeedSeparation = 1.0

    public init() {}
}

public struct NeedleEstimate: Codable, Sendable {
    public let model: String
    public let guidance: Float
    public let estimate: SubsetSimulation.Result
    /// Fraction of needle hits that stay hits under a fresh surrounding seed.
    public let backgroundIndependence: Double?
    /// Per facial feature: fraction of hits that survive resampling that feature's seed segment.
    public let featureRetention: [String: Double]
    /// Fraction of checked hits with a near-duplicate from a different seed (when configured).
    public var nearDuplicateRate: Double? = nil
}

public struct NeedleResult: Codable, Sendable {
    public struct DeltaBits: Codable, Sendable {
        public let guidance: Float
        public let value: Double
        /// Base never reached the needle: its μ is an upper bound, so Δ is a lower bound.
        public let isLowerBound: Bool
    }

    public struct AttackPoint: Codable, Sendable {
        public let guidance: Float
        public let attempts: Int
        public let target: Double
        public let base: Double
        /// The base's μ is an upper bound (it never reached the needle), so is this.
        public let baseIsBound: Bool
    }

    public struct Reachability: Codable, Sendable {
        public let trials: Int
        public let targetHitRate: Double
        public let baseHitRate: Double
    }

    public struct FootprintSummary: Codable, Sendable {
        public let label: String
        public let fraction: Double
        public let capturedInfluence: Double
        public let method: String
    }

    public let criterion: String
    public let threshold: Double
    public let thresholdMethod: String
    public let calibrationTPR: Double?
    public let calibrationFPR: Double?
    public let footprint: FootprintSummary?
    public let target: [NeedleEstimate]
    public let base: [NeedleEstimate]
    public let deltaBits: [DeltaBits]
    public let attackCurve: [AttackPoint]
    public let reachability: Reachability?
    public let earlyExitAgreement: Double?
    /// Mean χ² z-score of needle face segments inside the footprint (≈0: typical noise).
    public let needleChi2Z: Double?
    public let schedule: ScheduleInfo
    public let evaluations: Int
}

public struct NeedleProbe {
    public var configuration = NeedleConfiguration()

    public init(configuration: NeedleConfiguration = .init()) { self.configuration = configuration }

    /// Attack success within N attempts: 1 − (1 − μ)^N, stable for tiny μ.
    public static func attackProbability(_ mu: Double, attempts: Int) -> Double {
        guard mu > 0 else { return 0 }
        guard mu < 1 else { return 1 }
        return -expm1(Double(attempts) * log1p(-mu))
    }

    /// Fraction of samples with a partner that is a near-copy (masked squared latent distance
    /// ≤ `radius`) yet came from a clearly different seed (seed distance ≥ `separation`).
    /// Chains in subset simulation revisit nearby seeds; requiring distinct seeds keeps those
    /// from counting — what remains is different noise collapsing onto the same content.
    static func nearDuplicateRate(latents: MLXArray, seeds: MLXArray, region: MLXArray?, radius: Double,
                                  separation: Double) -> Double {
        let n = latents.dim(0)
        let w = (region.map { ($0 .> 0.5).asType(.float32) } ?? MLXArray.ones(Array(latents.shape.dropFirst())))
            .reshaped(1, -1)
        let count = max(w.sum().item(Float.self), 1)
        func pairwise(_ a: MLXArray) -> [Float] {
            let v = a.reshaped(n, -1) * w
            let sq = v.square().sum(axis: 1)
            return ((sq.reshaped(-1, 1) + sq.reshaped(1, -1) - 2 * matmul(v, v.transposed())) / count).asArray(Float.self)
        }
        let dx = pairwise(latents), dz = pairwise(seeds)
        var hits = 0
        for i in 0..<n where (0..<n).contains(where: { j in
            j != i && Double(dx[i * n + j]) <= radius && Double(dz[i * n + j]) >= separation
        }) {
            hits += 1
        }
        return Double(hits) / Double(n)
    }

    /// `prompt` overrides the attacker prompt (e.g. to measure capture by prompts that never
    /// ask for the person).
    public func run(target: DiffusionBackend, base: DiffusionBackend, reference x0: MLXArray,
                    criterion: NeedleCriterion, thresholdMethod: String, calibration: (tpr: Double, fpr: Double)? = nil,
                    footprint: SeedFootprint?, features: [SeedFootprint] = [], prompts: PromptSet,
                    prompt: String? = nil, schedule: SeedSchedule) throws -> NeedleResult {
        let config = configuration
        let shape = x0.shape
        let m = footprint?.mask
        let attacker = prompt ?? prompts.attacker
        let cond = try target.condition(prompt: attacker)
        let baseCond = try base.condition(prompt: attacker)
        let uncond = try target.condition(prompt: PromptSet.unconditional)
        let baseUncond = try base.condition(prompt: PromptSet.unconditional)

        func scorer(_ backend: DiffusionBackend, _ c: Conditioning, _ u: Conditioning, _ w: Float,
                    stop: Float? = nil) -> (MLXArray) -> [Double] {
            { z in
                criterion.scores(config.sampler.sample(backend, noise: z, conditions: [c], unconditional: u,
                                                      guidance: w, stopLogSNR: stop))
            }
        }

        var targets: [NeedleEstimate] = [], bases: [NeedleEstimate] = []
        var deltas: [NeedleResult.DeltaBits] = [], attack: [NeedleResult.AttackPoint] = []
        var evaluations = 0
        var needleChi2: Double?
        for (gi, w) in config.guidance.enumerated() {
            var pair: [NeedleEstimate] = []
            for (backend, c, u) in [(target, cond, uncond), (base, baseCond, baseUncond)] {
                let score = scorer(backend, c, u, w)
                // Same stream for target and base: paired random numbers.
                let (result, hits) = config.subset.run(shape: shape, footprint: m, target: criterion.threshold,
                                                       schedule: schedule, stream: gi + 1, score: score)
                evaluations += result.evaluations
                let checked = Array(hits.prefix(config.maxCheckedHits))
                var independence: Double?
                var retention: [String: Double] = [:]
                var duplicates: Double?
                if !checked.isEmpty {
                    let faces = stacked(checked.map(\.face)), backs = stacked(checked.map(\.background))
                    if let m {
                        var kept = 0, total = 0
                        for j in 0..<config.backgroundChecks {
                            let fresh = schedule.normal(.background, 900_000 + gi * 1000 + j, shape: [checked.count] + shape)
                            let s = score(SubsetSimulation.compose(faces, fresh, footprint: m.expandedDimensions(axis: 0)))
                            kept += s.filter { $0 >= criterion.threshold }.count
                            total += s.count
                        }
                        evaluations += total
                        independence = Double(kept) / Double(max(total, 1))
                    }
                    let z = SubsetSimulation.compose(faces, backs, footprint: m?.expandedDimensions(axis: 0))
                    for feature in features {
                        let mf = feature.mask.expandedDimensions(axis: 0)
                        let fresh = schedule.normal(.jitter, 910_000 + gi, shape: [checked.count] + shape)
                        let s = score(sqrt(1 - mf.square()) * z + mf * fresh)
                        evaluations += s.count
                        retention[feature.label] = Double(s.filter { $0 >= criterion.threshold }.count) / Double(s.count)
                    }
                    if let radius = config.duplicateRadius, checked.count >= 2 {
                        let latents = config.sampler.sample(backend, noise: z, conditions: [c], unconditional: u, guidance: w)
                        evaluations += checked.count
                        duplicates = Self.nearDuplicateRate(latents: latents, seeds: faces, region: m, radius: radius,
                                                            separation: config.duplicateSeedSeparation)
                    }
                    if backend === target, gi == 0, let m {
                        let region = m .> 0.5
                        needleChi2 = checked.map {
                            InversionProbe.typicality($0.face, mask: region, label: "needle", spectral: false).chi2Z
                        }.reduce(0, +) / Double(checked.count)
                    }
                }
                pair.append(NeedleEstimate(model: backend.identifier, guidance: w, estimate: result,
                                           backgroundIndependence: independence, featureRetention: retention,
                                           nearDuplicateRate: duplicates))
            }
            targets.append(pair[0])
            bases.append(pair[1])
            let t = pair[0].estimate, b = pair[1].estimate
            deltas.append(.init(guidance: w, value: log2(max(t.probability, 1e-300)) - log2(max(b.probability, 1e-300)),
                                isLowerBound: !b.reachedTarget))
            for n in config.attackAttempts {
                attack.append(.init(guidance: w, attempts: n, target: Self.attackProbability(t.probability, attempts: n),
                                    base: Self.attackProbability(b.probability, attempts: n), baseIsBound: !b.reachedTarget))
            }
        }

        // Reachability: the inverted reference seed's face segment, with fresh surroundings.
        var fine = DeterministicSampler(solver: .ddim, steps: config.reachabilitySteps)
        fine.lambdaMin = config.sampler.lambdaMin
        fine.lambdaMax = config.sampler.lambdaMax
        let trials = 8
        func reach(_ backend: DiffusionBackend, _ c: Conditioning) -> Double {
            let zStar = fine.invert(backend, x0: x0.expandedDimensions(axis: 0), conditions: [c])
            let faces = broadcast(zStar, to: [trials] + shape)
            let fresh = schedule.normal(.background, 950_000, shape: [trials] + shape)
            let z = SubsetSimulation.compose(faces, fresh, footprint: m?.expandedDimensions(axis: 0))
            let s = criterion.scores(fine.sample(backend, noise: z, conditions: [c]))
            return Double(s.filter { $0 >= criterion.threshold }.count) / Double(trials)
        }
        let reachability = config.checkReachability
            ? NeedleResult.Reachability(trials: trials, targetHitRate: reach(target, cond), baseHitRate: reach(base, baseCond))
            : nil

        // Early exit: do decisions at λ_stop agree with full sampling?
        var agreement: Double?
        if let stop = config.earlyExitLogSNR {
            let z = schedule.normal(.needle, 990_000, shape: [100] + shape)
            let full = scorer(target, cond, uncond, config.guidance[0])(z)
            let early = scorer(target, cond, uncond, config.guidance[0], stop: stop)(z)
            let same = zip(full, early).filter { ($0.0 >= criterion.threshold) == ($0.1 >= criterion.threshold) }.count
            agreement = Double(same) / Double(full.count)
            evaluations += 200
        }

        return NeedleResult(
            criterion: criterion.name, threshold: criterion.threshold, thresholdMethod: thresholdMethod,
            calibrationTPR: calibration?.tpr, calibrationFPR: calibration?.fpr,
            footprint: footprint.map { .init(label: $0.label, fraction: $0.fraction, capturedInfluence: $0.capturedInfluence,
                                             method: $0.method.rawValue) },
            target: targets, base: bases, deltaBits: deltas, attackCurve: attack, reachability: reachability,
            earlyExitAgreement: agreement, needleChi2Z: needleChi2, schedule: schedule.info, evaluations: evaluations)
    }
}
