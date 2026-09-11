//
//  MemoryPoolProbe.swift
//  ScorpionKit
//
//  Is the reference in what the model memorized? Working backwards from Kim & Lee (arXiv
//  2605.26756): memorization is coordinate-wise variance collapse — the model stops letting
//  some coordinates vary — visible as curvature against an underfitted baseline. A memory
//  pool is where that collapse lives: an attractor of sampling (Biroli et al. 2024) that
//  pulls trajectories onto stored content. Four questions, all decode-free:
//
//  1. Is there a pool here?  Retained variance ρ = 1 − σ·diag(∂ε̂/∂x) at the *noised
//     reference*, across noise levels; collapse = ρ_base − ρ_target (what the weights pin)
//     and ρ_uncond − ρ_cond (what the prompt pins), per region and as a latent-grid map.
//  2. Is it yours?  Where the target denoises the noised reference: back onto the photo
//     itself (snap = base distance / target distance) and/or onto the person (identity
//     criterion) — or onto someone else's pool nearby.
//  3. Can a seed reach it?  Invert the reference to its seed endpoint z*, move the face
//     segment by β (pCN), resample the surroundings, and count returns: r(0) is
//     reachability, r(1) is the needle mass, β½ is the pool's radius in seed space.
//  4. Does it take over other generations?  Needle mass under trigger, descriptor and empty
//     prompts, and how often hits recur as near-duplicates from different seeds.
//
//  One photo is enough for questions 1–3: memorization stores points, not concepts. A
//  well-generalized identity leaves no localized collapse (the paper's own limit) — that
//  case is the needle's (Tier L), which this probe never replaces.
//

import Foundation
import MLX

public struct MemoryPoolConfiguration: Codable, Sendable {
    /// Noise levels (log-SNR) the reference is probed at, noisy → clean.
    public var logSNRs: [Float] = [-4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    public var noiseDraws = 8
    /// Hutchinson probes per model and prompt; 0 = forward-only score differences.
    public var probes = 16
    /// Face collapse (ρ_base − ρ_target) that counts as a pool.
    public var collapseThreshold = 0.1
    /// Only noise levels where the base has *released* the face (its retained variance ≥ this)
    /// count toward a pool: memorization pins what the base leaves free. At higher noise a base
    /// that doesn't know the face hesitates between faces — extra variance on the base side
    /// that would read as collapse for any model that merely learned the face.
    public var releaseThreshold = 0.5
    /// …and only levels where the base keeps a single face: within one mode retained variance
    /// is at most 1, so ρ_base (or its forward spread) above this means the base is choosing
    /// *between* faces — the same confound at lower noise. (The paper's remedy is a baseline
    /// that already knows the face: a less-trained checkpoint or the unconditional branch;
    /// `promptCollapse` is that comparison whenever a prompt gates the memory.)
    public var mixingLimit = 1.1
    /// The target must return the noised photo to itself this many times tighter than the base.
    public var snapThreshold = 4.0
    /// Share of the person's generations recurring as near-copies that marks stored photos.
    public var recurrenceThreshold = 0.25
    public var basinBetas: [Float] = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1]
    public var basinTrials = 32
    /// Inversion and resampling share one fine grid (the pre-image is sampler-specific).
    public var basinSampler = DeterministicSampler(solver: .ddim, steps: 128)
    public var checkBasin = true
    public var capture = true
    /// Prompt kinds capture is measured under ("trigger" applies only when the model names triggers).
    public var captureKinds = ["trigger", "descriptor", "empty"]
    /// Needle settings for capture (reachability and early exit are not needed there).
    public var needle = NeedleConfiguration()

    public init() {
        needle.checkReachability = false
        needle.earlyExitLogSNR = nil
        needle.backgroundChecks = 0
    }
}

public struct MemoryPoolResult: Codable, Sendable {
    public struct Split: Codable, Sendable {
        public let face: Double
        public let background: Double
    }

    public struct Point: Codable, Sendable {
        public let logSNR: Float
        /// Retained variance ρ (nil in forward-only mode).
        public let retainedTarget: Split?
        public let retainedBase: Split?
        public let retainedUnconditional: Split?
        /// ρ_base − ρ_target: what the weights pin that the base leaves free.
        public let collapse: Split?
        /// ρ_uncond − ρ_cond under the target: what the prompt pins.
        public let promptCollapse: Split?
        /// (ε̂_target − ε̂_base)², forward-only.
        public let scoreDifference: Split
        /// (ε̂_cond − ε̂_uncond)² under the target, forward-only.
        public let promptScoreDifference: Split?
        /// Forward-only retained variance: ρ̂ = sd_ε(x̂₀)·α/σ across the noise draws. Tracks ρ
        /// where the model's answer is locally linear in the noise (a pool at the reference);
        /// where it switches between pools the switching reads as extra spread.
        public let spreadTarget: Split
        public let spreadBase: Split
        /// ρ̂_base − ρ̂_target.
        public let spreadCollapse: Split
        /// Masked mean squared distance from the denoised estimate x̂₀ to the reference.
        public let selfDistanceTarget: Double
        public let selfDistanceBase: Double
        /// Fraction of denoised estimates the identity criterion accepts as the person.
        public let identityTarget: Double?
        public let identityBase: Double?
        /// IoU of the map's top positions (as many as the face has) with the face.
        public let iou: Double

        /// How many times tighter the target returns the noised photo to itself than the base.
        public var snap: Double { selfDistanceBase / max(selfDistanceTarget, 1e-12) }
        /// The pool signal: face collapse (Hutchinson), or its forward-only estimate.
        public var signal: Double { collapse?.face ?? spreadCollapse.face }
        public var backgroundSignal: Double { collapse?.background ?? spreadCollapse.background }
        /// The base's retained variance on the face (how free the base leaves it here).
        public var baseRetained: Double { retainedBase?.face ?? spreadBase.face }
    }

    public struct Basin: Codable, Sendable {
        public let betas: [Float]
        public let trials: Int
        /// Return rate r(β) — the fraction of perturbed seeds that come back to the reference.
        public let target: [Double]
        public let base: [Double]
        /// Seed-space radius where r falls to ½ (0: never robustly reached; 1: survives any move).
        public let halfRadiusTarget: Float
        public let halfRadiusBase: Float
        /// χ² z-score of the inverted seed's face segment (≈ 0: a typical seed).
        public let seedChi2ZTarget: Double
        public let seedChi2ZBase: Double
        public let sampler: String
    }

    public struct Capture: Codable, Sendable {
        public let kind: String
        public let prompt: String
        public let target: Double
        /// False when the target never reached the face: `target` is then a bound.
        public let targetReached: Bool
        public let base: Double
        public let baseIsBound: Bool
        public let deltaBits: Double
        /// Fraction of the target's hits that recur as near-copies from different seeds.
        public let nearDuplicateRate: Double?
    }

    public enum Tier: String, Codable, Sendable {
        /// A pool at the reference that returns the photo to itself: the photo (or a near
        /// copy) is stored in the weights.
        case memorizedPhoto = "memorized-photo"
        /// A pool that pulls the reference onto the same person: other photos of this face
        /// are stored.
        case identityPool = "identity-pool"
        /// A pool near the reference that pulls it onto someone else.
        case nearbyPool = "nearby-pool"
        /// No localized collapse, but the face is generable (needle Δbits).
        case generalizedLikeness = "generalized-likeness"
        case noEvidence = "no-evidence"
    }

    public let method: String
    public let prompt: String
    public let points: [Point]
    /// λ span where face collapse ≥ threshold (at levels where the base has released the face).
    public let collapseBand: [Float]?
    public let peakLogSNR: Float
    /// The pool signal: face collapse sustained over two adjacent released levels.
    public let peakSignal: Double
    public let peakSnap: Double
    /// Geometric-mean snap over the levels where the base has released the face — the
    /// single-photo membership statistic (it doesn't depend on where a peak happens to fall).
    public let bandSnap: Double
    /// Face share of the collapse at the peak: face / (face + background).
    public let localization: Double
    public let iou: Double
    /// Share of the face's collapse mass inside each feature.
    public let featureShare: [String: Double]
    /// Collapse map at the peak over the latent grid (channels summed; noise space statistics).
    public let map: [Float]
    public let mapShape: [Int]
    public let basin: Basin?
    public let capture: [Capture]
    public let tier: Tier
    public let reasons: [String]
    public let calibrated: Bool
    public let schedule: ScheduleInfo
    /// Denoiser evaluations (a VJP counts once).
    public let evaluations: Int
}

public struct MemoryPoolProbe {
    public var configuration = MemoryPoolConfiguration()

    public init(configuration: MemoryPoolConfiguration = .init()) { self.configuration = configuration }

    /// The noised reference at λ: x_λ = α·x₀ + σ·ε for `n` keyed draws (shared by every model).
    public static func noised(_ backend: DiffusionBackend, x0: MLXArray, logSNR: Float, index: Int, draws: Int,
                              schedule: SeedSchedule) -> (x: MLXArray, alpha: Float, sigma: Float) {
        let (a, s) = backend.alphaSigma(logSNR: MLXArray([logSNR]))
        let alpha = a.item(Float.self), sigma = s.item(Float.self)
        let eps = schedule.normal(.noise, 800_000 + index, shape: [draws] + x0.shape)
        return (alpha * x0.expandedDimensions(axis: 0) + sigma * eps, alpha, sigma)
    }

    /// - Parameters:
    ///   - reference: the reference latent (latent shape).
    ///   - face: face weights (latent shape).
    ///   - identity: calibrated "is this the person" criterion (optional).
    ///   - photo: "is this the photo" criterion for the basin (optional: no basin without it).
    public func run(target: DiffusionBackend, base: DiffusionBackend, reference x0: MLXArray, face: MLXArray,
                    features: [SeedFootprint] = [], identity: NeedleCriterion? = nil, photo: NeedleCriterion? = nil,
                    footprint: SeedFootprint?, prompts: PromptSet, schedule: SeedSchedule) throws -> MemoryPoolResult {
        let config = configuration
        let shape = x0.shape
        let n = config.noiseDraws
        let prompt = prompts.attacker
        let condT = try target.condition(prompt: prompt), condB = try base.condition(prompt: prompt)
        let uncT = try target.condition(prompt: PromptSet.unconditional)
        let pinned = !prompt.isEmpty
        let faceW = face.reshaped([1] + shape), backW = 1 - faceW
        let ref = x0.expandedDimensions(axis: 0)
        let h = shape[shape.count - 2], w = shape[shape.count - 1]
        let faceGrid = (face.reshaped(-1, h, w).max(axis: 0) .> 0.5).reshaped(-1).asArray(Bool.self)
        var evaluations = 0

        func mean(_ m: MLXArray, _ weights: MLXArray) -> Double {
            Double(((m * weights).sum() / (weights.sum() * Float(m.dim(0)) + 1e-12)).item(Float.self))
        }
        func split(_ m: MLXArray) -> MemoryPoolResult.Split { .init(face: mean(m, faceW), background: mean(m, backW)) }
        func accepted(_ c: NeedleCriterion?, _ x: MLXArray) -> Double? {
            c.map { crit in Double(crit.scores(x).filter { $0 >= crit.threshold }.count) / Double(x.dim(0)) }
        }

        // 1–2. Collapse and destination at the noised reference, level by level.
        var points: [MemoryPoolResult.Point] = []
        var maps: [[Float]] = []
        for (li, lambda) in config.logSNRs.enumerated() {
            let l = MLXArray([Float](repeating: lambda, count: n))
            let (xt, alpha, sigma) = Self.noised(target, x0: x0, logSNR: lambda, index: li, draws: n, schedule: schedule)
            let eT = target.predictEpsilon(xt, logSNR: l, conditions: [condT])
            let eB = base.predictEpsilon(xt, logSNR: l, conditions: [condB])
            let eU = pinned ? target.predictEpsilon(xt, logSNR: l, conditions: [uncT]) : nil
            evaluations += n * (pinned ? 3 : 2)
            let x0T = (xt - sigma * eT) / alpha, x0B = (xt - sigma * eB) / alpha
            // Forward-only ρ̂: how much the denoised estimate moves with the noise.
            func spread(_ x: MLXArray) -> MLXArray {
                broadcast(sqrt(x.variance(axis: 0, keepDims: true, ddof: 1)) * (alpha / sigma), to: x.shape)
            }
            let sT = spread(x0T), sB = spread(x0B)

            var rT: MemoryPoolResult.Split?, rB: MemoryPoolResult.Split?, rU: MemoryPoolResult.Split?
            var collapse: MemoryPoolResult.Split?, promptCollapse: MemoryPoolResult.Split?
            var map: MLXArray
            if config.probes > 0 {
                let sig = MLXArray([Float](repeating: sigma, count: n))
                // Same probe vectors for every model and prompt: the differences are paired.
                func rho(_ b: DiffusionBackend, _ c: Conditioning) -> MLXArray {
                    Curvature.retainedVariance(
                        diagJacobian: Curvature.diagJacobian(b, x: xt, logSNR: l, conditions: [c], probes: config.probes,
                                                             schedule: schedule, index: 700 + li),
                        sigma: sig)
                }
                let pT = rho(target, condT), pB = rho(base, condB)
                let pU = pinned ? rho(target, uncT) : nil
                evaluations += n * config.probes * (pinned ? 3 : 2)
                rT = split(pT)
                rB = split(pB)
                rU = pU.map(split)
                collapse = split(pB - pT)
                promptCollapse = pU.map { split($0 - pT) }
                map = pB - pT
            } else {
                map = sB - sT
            }
            let grid = map.mean(axis: 0).reshaped(-1, h, w).sum(axis: 0).reshaped(-1).asArray(Float.self)
            maps.append(grid)
            points.append(.init(
                logSNR: lambda, retainedTarget: rT, retainedBase: rB, retainedUnconditional: rU,
                collapse: collapse, promptCollapse: promptCollapse,
                scoreDifference: split(Curvature.scoreDifference(eT, eB)),
                promptScoreDifference: eU.map { split(Curvature.scoreDifference(eT, $0)) },
                spreadTarget: split(sT), spreadBase: split(sB), spreadCollapse: split(sB - sT),
                selfDistanceTarget: mean((x0T - ref).square(), faceW), selfDistanceBase: mean((x0B - ref).square(), faceW),
                identityTarget: accepted(identity, x0T), identityBase: accepted(identity, x0B),
                iou: Self.topIoU(grid, faceGrid)))
        }

        let released = points.indices.filter {
            let p = points[$0]
            return p.baseRetained >= config.releaseThreshold && p.baseRetained <= config.mixingLimit
                && p.spreadBase.face <= config.mixingLimit
        }
        let candidates = released.isEmpty ? Array(points.indices) : released
        // A pool pins the face over a *band* of noise levels — from the base's natural
        // variability down to the pool's width — while a base hesitating between faces does so
        // in one narrow window. So the pool signal is the collapse sustained over two adjacent
        // released levels (single levels only when no two are adjacent).
        let pairs = zip(candidates, candidates.dropFirst()).filter { $0.1 == $0.0 + 1 }
        var sustained = -Double.infinity, peak = candidates.first ?? 0
        if pairs.isEmpty {
            for i in candidates where points[i].signal > sustained { sustained = points[i].signal; peak = i }
        } else {
            for (i, j) in pairs where min(points[i].signal, points[j].signal) > sustained {
                sustained = min(points[i].signal, points[j].signal)
                peak = points[i].signal >= points[j].signal ? i : j
            }
        }
        let above = Set(candidates.filter { points[$0].signal >= config.collapseThreshold })
        let band = above.filter { above.contains($0 - 1) || above.contains($0 + 1) || pairs.isEmpty }
            .sorted().map { points[$0].logSNR }
        let bandSnap = exp(candidates.map { log(max(points[$0].snap, 1e-12)) }.reduce(0, +) / Double(max(candidates.count, 1)))
        let peakMap = maps[peak]
        let peakPoint = points[peak]
        let faceMass = zip(peakMap, faceGrid).reduce(0.0) { $0 + ($1.1 ? Double(max($1.0, 0)) : 0) }
        var shares: [String: Double] = [:]
        for feature in features {
            let fg = (feature.mask.reshaped(-1, h, w).max(axis: 0) .> 0.5).reshaped(-1).asArray(Bool.self)
            let mass = zip(peakMap, fg).reduce(0.0) { $0 + ($1.1 ? Double(max($1.0, 0)) : 0) }
            shares[feature.label] = faceMass > 0 ? mass / faceMass : 0
        }
        let faceSignal = max(peakPoint.signal, 0), backSignal = max(peakPoint.backgroundSignal, 0)
        let localization = faceSignal / max(faceSignal + backSignal, 1e-12)

        // 3. Basin around the reference's seed endpoint.
        var basin: MemoryPoolResult.Basin?
        if config.checkBasin, let photo {
            let fine = config.basinSampler
            let m = footprint?.mask.reshaped([1] + shape)
            let region = face .> 0.5
            func curve(_ backend: DiffusionBackend, _ c: Conditioning) -> ([Double], Double) {
                let zStar = fine.invert(backend, x0: ref, conditions: [c])
                let rates = config.basinBetas.enumerated().map { bi, beta -> Double in
                    let xi = schedule.normal(.basin, 1_000 + bi, shape: [config.basinTrials] + shape)
                    let fresh = schedule.normal(.background, 970_000 + bi, shape: [config.basinTrials] + shape)
                    let z = SubsetSimulation.compose(sqrt(1 - beta * beta) * zStar + beta * xi, fresh, footprint: m)
                    let s = photo.scores(fine.sample(backend, noise: z, conditions: [c]))
                    return Double(s.filter { $0 >= photo.threshold }.count) / Double(s.count)
                }
                let chi2 = InversionProbe.typicality(zStar.squeezed(axis: 0), mask: region, label: "z*", spectral: false).chi2Z
                return (rates, chi2)
            }
            let (rt, ct) = curve(target, condT), (rb, cb) = curve(base, condB)
            evaluations += 2 * fine.steps * (1 + config.basinBetas.count * config.basinTrials)
            basin = .init(betas: config.basinBetas, trials: config.basinTrials, target: rt, base: rb,
                          halfRadiusTarget: Self.halfRadius(config.basinBetas, rt),
                          halfRadiusBase: Self.halfRadius(config.basinBetas, rb),
                          seedChi2ZTarget: ct, seedChi2ZBase: cb, sampler: "\(fine.solver.rawValue)-\(fine.steps)")
        }

        // 4. Capture: does the face take over generations, including prompts that never ask for it?
        var capture: [MemoryPoolResult.Capture] = []
        if config.capture, let criterion = identity ?? photo {
            var kinds: [(String, String)] = []
            if !prompts.triggers.isEmpty { kinds.append(("trigger", prompts.attacker)) }
            kinds.append(("descriptor", prompts.descriptor))
            kinds.append(("empty", PromptSet.unconditional))
            kinds = kinds.filter { config.captureKinds.contains($0.0) }
            let probe = NeedleProbe(configuration: config.needle)
            let perSeed = config.needle.sampler.steps * ((config.needle.guidance.first ?? 1) == 1 ? 1 : 2)
            for (kind, p) in kinds {
                let r = try probe.run(target: target, base: base, reference: x0, criterion: criterion,
                                      thresholdMethod: "memory-capture", footprint: footprint, prompts: prompts,
                                      prompt: p, schedule: schedule)
                evaluations += r.evaluations * perSeed
                capture.append(.init(kind: kind, prompt: p, target: r.target[0].estimate.probability,
                                     targetReached: r.target[0].estimate.reachedTarget,
                                     base: r.base[0].estimate.probability, baseIsBound: r.deltaBits[0].isLowerBound,
                                     deltaBits: r.deltaBits[0].value, nearDuplicateRate: r.target[0].nearDuplicateRate))
            }
        }

        let (tier, reasons) = Self.tier(sustained: sustained, peak: peakPoint, basin: basin, capture: capture, config: config)
        return MemoryPoolResult(
            method: config.probes > 0 ? "hutchinson-\(config.probes)" : "forward-spread", prompt: prompt,
            points: points, collapseBand: band.isEmpty ? nil : [band.min()!, band.max()!],
            peakLogSNR: peakPoint.logSNR, peakSignal: sustained, peakSnap: peakPoint.snap, bandSnap: bandSnap,
            localization: localization, iou: peakPoint.iou, featureShare: shares, map: peakMap, mapShape: [h, w],
            basin: basin, capture: capture, tier: tier, reasons: reasons, calibrated: false,
            schedule: schedule.info, evaluations: evaluations)
    }

    /// Rule-based readout (uncalibrated, like every Scorpion score until labeled data exists).
    static func tier(sustained: Double, peak: MemoryPoolResult.Point, basin: MemoryPoolResult.Basin?,
                     capture: [MemoryPoolResult.Capture], config: MemoryPoolConfiguration)
        -> (MemoryPoolResult.Tier, [String]) {
        var reasons: [String] = []
        let pool = sustained >= config.collapseThreshold
        if pool {
            reasons.append(String(format: "face collapse %.2f sustained around λ = %.0f (threshold %.2f)", sustained, peak.logSNR,
                                  config.collapseThreshold))
        } else {
            reasons.append(String(format: "no sustained localized collapse (%.2f < %.2f)", sustained, config.collapseThreshold))
        }
        let snaps = peak.snap >= config.snapThreshold
        let wider = basin.map { $0.halfRadiusTarget > $0.halfRadiusBase } ?? true
        // Recurrence: the person's generations come back as near-copies from different seeds —
        // stored photos of this face, even when the reference itself is not one of them.
        let recurring = capture.first { $0.targetReached && ($0.nearDuplicateRate ?? 0) >= config.recurrenceThreshold }
        func recurrence(_ c: MemoryPoolResult.Capture) -> String {
            String(format: "%.0f%% of the person's generations (%@ prompt) recur as near-copies from different seeds",
                   100 * (c.nearDuplicateRate ?? 0), c.kind)
        }
        if pool {
            if peak.snap >= 1 {
                reasons.append(String(format: "denoises the photo back onto itself %.1f× tighter than the base", peak.snap))
            } else {
                reasons.append(String(format: "pulls the noised photo onto a different stored face (%.1f× farther than the base)",
                                      1 / max(peak.snap, 1e-9)))
            }
            if let basin {
                reasons.append(String(format: "seed-endpoint radius β½ %.2f vs base %.2f", basin.halfRadiusTarget, basin.halfRadiusBase))
            }
            if snaps && wider { return (.memorizedPhoto, reasons) }
            if let id = peak.identityTarget {
                reasons.append(String(format: "%.0f%% of pulled estimates are the person (base %.0f%%)", 100 * id,
                                      100 * (peak.identityBase ?? 0)))
                if id >= 0.5 { return (.identityPool, reasons) }
            }
            if let recurring {
                reasons.append(recurrence(recurring))
                return (.identityPool, reasons)
            }
            return (.nearbyPool, reasons)
        }
        if let recurring {
            reasons.append(recurrence(recurring))
            return (.identityPool, reasons)
        }
        if let lead = capture.first, lead.targetReached, lead.deltaBits >= 3 {
            reasons.append(String(format: "but the face is generable: %+.1f bits under the %@ prompt, without recurring copies",
                                  lead.deltaBits, lead.kind))
            return (.generalizedLikeness, reasons)
        }
        return (.noEvidence, reasons)
    }

    /// IoU of the top-|face| positions of `map` with the face.
    static func topIoU(_ map: [Float], _ face: [Bool]) -> Double {
        let k = face.filter { $0 }.count
        guard k > 0, k < face.count else { return 0 }
        let top = Set(map.indices.sorted { map[$0] > map[$1] }.prefix(k))
        let inter = top.filter { face[$0] }.count
        return Double(inter) / Double(2 * k - inter)
    }

    /// Median masked squared latent distance between photos of one set — the identity's
    /// photo-to-photo scale, which sets the photo-level and near-duplicate radii. Needs ≥ 2 photos.
    public static func photoScale(_ set: MLXArray, mask: MLXArray) -> Double? {
        let n = set.dim(0)
        guard n >= 2 else { return nil }
        let w = mask.reshaped(1, -1)
        let v = set.reshaped(n, -1) * w
        let sq = v.square().sum(axis: 1)
        let d = ((sq.reshaped(-1, 1) + sq.reshaped(1, -1) - 2 * matmul(v, v.transposed())) / w.sum()).asArray(Float.self)
        let pairs = (0..<n).flatMap { i in ((i + 1)..<n).map { Double(d[i * n + $0]) } }.sorted()
        return pairs[pairs.count / 2]
    }

    /// "Is this the photo": within a quarter of the photo-to-photo scale of the reference.
    public static func photoCriterion(reference x0: MLXArray, mask: MLXArray, scale: Double) -> LatentFidelity {
        LatentFidelity(references: x0.expandedDimensions(axis: 0), mask: mask, threshold: -0.25 * scale, aggregate: .nearest)
    }

    /// First β where r(β) falls below ½, linearly interpolated (0 if r(0) < ½; 1 if never).
    static func halfRadius(_ betas: [Float], _ r: [Double]) -> Float {
        guard let first = r.first, first >= 0.5 else { return 0 }
        for i in 1..<r.count where r[i] < 0.5 {
            let t = Float((r[i - 1] - 0.5) / max(r[i - 1] - r[i], 1e-12))
            return betas[i - 1] + t * (betas[i] - betas[i - 1])
        }
        return 1
    }
}
