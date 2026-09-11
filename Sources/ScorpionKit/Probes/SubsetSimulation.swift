//
//  SubsetSimulation.swift
//  ScorpionKit
//
//  Rare-event probability under a standard Gaussian: μ = P_{z∼N(0,I)}(g(z) ≥ t) — "a needle
//  in a haystack". Plain Monte Carlo needs ~1/μ samples; subset simulation (Au & Beck 2001)
//  writes μ as a product of conditional probabilities P(g ≥ t_{l+1} | g ≥ t_l) ≈ p0 each,
//  sampling every level with MCMC. Moves use preconditioned Crank–Nicolson (Cotter et al.
//  2013), z' = √(1−β²)·z + β·ξ, which leaves N(0,I) invariant in any dimension.
//
//  The seed is split into a face segment and a background: z = m⊙z_F + √(1−m²)⊙z_B keeps
//  every element N(0,1). Chains alternate a pCN move on z_F with an independent fresh draw
//  of z_B (accepted iff still above threshold) — a valid Metropolis step whose acceptance
//  rate measures whether "the surrounding seed can be anything".
//

import Foundation
import MLX

public struct SubsetSimulation: Codable, Sendable {
    public var samplesPerLevel = 200
    public var levelProbability = 0.1
    public var maxLevels = 6
    public var initialBeta: Float = 0.5
    public var resampleBackground = true
    /// Scoring batch size.
    public var chunk = 100

    public init() {}

    public struct Result: Codable, Sendable {
        /// μ̂, or an upper bound when `reachedTarget` is false.
        public let probability: Double
        public let reachedTarget: Bool
        /// Chains stopped making progress (thresholds stopped rising): `probability` is a bound.
        public let stalled: Bool
        public let levels: Int
        public let thresholds: [Double]
        /// Au & Beck coefficient of variation of μ̂ (includes within-chain correlation).
        public let coefficientOfVariation: Double
        public let evaluations: Int
        public let pcnAcceptance: [Double]
        public let backgroundAcceptance: [Double]

        /// Bits of luck an attacker needs: −log₂ μ.
        public var bits: Double { -log2(max(probability, 1e-300)) }
        /// 90% interval on log₂ μ (delta method).
        public var log2CI90: [Double] {
            let center = log2(max(probability, 1e-300))
            let half = 1.645 * coefficientOfVariation / log(2.0)
            return [center - half, center + half]
        }
    }

    public struct Sample {
        public let face: MLXArray
        public let background: MLXArray
        public let score: Double
    }

    /// Compose a seed from its face segment and background with soft footprint `m`.
    public static func compose(_ face: MLXArray, _ background: MLXArray, footprint m: MLXArray?) -> MLXArray {
        guard let m else { return face }
        return m * face + sqrt(1 - m.square()) * background
    }

    /// `shape`: latent shape. `score`: batched seeds (B, …shape) → scores.
    /// `stream`: offset keeping independent runs apart; runs with the same offset share
    /// random numbers (paired comparisons).
    public func run(shape: [Int], footprint: MLXArray?, target: Double, schedule: SeedSchedule, stream: Int,
                    score: (MLXArray) -> [Double]) -> (Result, [Sample]) {
        let n = samplesPerLevel
        let nc = max(1, Int((levelProbability * Double(n)).rounded()))
        let ns = max(1, n / nc)
        let m = footprint.map { $0.reshaped([1] + shape) }
        let base = stream * 1_000_000

        func scoreAll(_ z: MLXArray) -> [Double] {
            var out: [Double] = []
            var i = 0
            while i < z.dim(0) {
                out += score(z[i..<min(i + chunk, z.dim(0))])
                i += chunk
            }
            return out
        }

        var zF = schedule.normal(.needle, base, shape: [n] + shape)
        var zB = schedule.normal(.background, base, shape: [n] + shape)
        var g = scoreAll(Self.compose(zF, zB, footprint: m))
        var evaluations = n
        var chains: Int? = nil          // nil: iid level; else samples are step-major over `chains`
        var prefix = 1.0
        var deltaSq = 0.0
        var thresholds: [Double] = []
        var pcnAcc: [Double] = [], bgAcc: [Double] = []
        var beta = initialBeta
        var level = 0

        while true {
            let order = g.indices.sorted { g[$0] > g[$1] }
            let k = min(nc, g.count - 1)
            let next = (g[order[k - 1]] + g[order[k]]) / 2
            if next < target, let last = thresholds.last, next <= last + 1e-9 * max(1, abs(last)) {
                // No progress: every chain is stuck. Report what we know — a bound.
                let result = Result(probability: prefix, reachedTarget: false, stalled: true, levels: level,
                                    thresholds: thresholds, coefficientOfVariation: deltaSq.squareRoot(),
                                    evaluations: evaluations, pcnAcceptance: pcnAcc, backgroundAcceptance: bgAcc)
                return (result, [])
            }
            if next >= target || level >= maxLevels {
                let hitIdx = g.indices.filter { g[$0] >= target }
                let pF = Double(hitIdx.count) / Double(g.count)
                let probability: Double
                if pF > 0 {
                    probability = prefix * pF
                    let gamma = chains.map { Self.gamma(g.map { $0 >= target }, chains: $0) } ?? 0
                    deltaSq += (1 - pF) / (pF * Double(g.count)) * (1 + gamma)
                } else {
                    probability = prefix * 3 / Double(g.count)   // ~95% upper bound on the last factor
                }
                let samples = hitIdx.map { Sample(face: zF[$0], background: zB[$0], score: g[$0]) }
                let result = Result(probability: probability, reachedTarget: pF > 0, stalled: false, levels: level,
                                    thresholds: thresholds, coefficientOfVariation: deltaSq.squareRoot(),
                                    evaluations: evaluations, pcnAcceptance: pcnAcc, backgroundAcceptance: bgAcc)
                return (result, samples)
            }

            // Intermediate level: condition on g ≥ next.
            let gamma = chains.map { Self.gamma(g.map { $0 >= next }, chains: $0) } ?? 0
            let pl = Double(k) / Double(g.count)
            deltaSq += (1 - pl) / (pl * Double(g.count)) * (1 + gamma)
            prefix *= pl
            thresholds.append(next)

            let seeds = MLXArray(order.prefix(k).map { Int32($0) })
            var cF = zF[seeds], cB = zB[seeds]
            var cG = order.prefix(k).map { g[$0] }
            var facesOut = [cF], backsOut = [cB], scoresOut = cG
            var accepted = 0, bgAccepted = 0
            let bshape = [k] + Array(repeating: 1, count: shape.count)
            var logBeta = log(Double(beta))
            for step in 1..<ns {
                let key = base + (level + 1) * 1000 + step
                let b = Float(exp(logBeta))
                let prop = sqrt(1 - b * b) * cF + b * schedule.normal(.needle, key, shape: [k] + shape)
                let gp = scoreAll(Self.compose(prop, cB, footprint: m))
                evaluations += k
                let ok = gp.map { $0 >= next }
                let stepAccepted = ok.filter { $0 }.count
                accepted += stepAccepted
                // Robbins–Monro adaptation of the pCN step toward ~44% acceptance, within the
                // level: thin conditional sets need much smaller steps than the prior.
                logBeta += (Double(stepAccepted) / Double(k) - 0.44) / Double(step).squareRoot()
                logBeta = min(0, max(log(0.002), logBeta))
                cF = which(MLXArray(ok).reshaped(bshape), prop, cF)
                cG = zip(zip(ok, gp), cG).map { $0.0 ? $0.1 : $1 }
                if resampleBackground, m != nil {
                    let fresh = schedule.normal(.background, key, shape: [k] + shape)
                    let gb = scoreAll(Self.compose(cF, fresh, footprint: m))
                    evaluations += k
                    let okB = gb.map { $0 >= next }
                    bgAccepted += okB.filter { $0 }.count
                    cB = which(MLXArray(okB).reshaped(bshape), fresh, cB)
                    cG = zip(zip(okB, gb), cG).map { $0.0 ? $0.1 : $1 }
                }
                eval(cF, cB)
                facesOut.append(cF)
                backsOut.append(cB)
                scoresOut += cG
            }
            let tries = Double(max(k * (ns - 1), 1))
            pcnAcc.append(Double(accepted) / tries)
            if resampleBackground, m != nil { bgAcc.append(Double(bgAccepted) / tries) }
            beta = Float(exp(logBeta))   // next level starts from the adapted step

            zF = concatenated(facesOut, axis: 0)
            zB = concatenated(backsOut, axis: 0)
            g = scoresOut
            chains = k
            level += 1
        }
    }

    /// Within-chain correlation factor γ for indicator samples laid out step-major over
    /// `chains` chains (Au & Beck 2001, eq. 29).
    static func gamma(_ indicators: [Bool], chains: Int) -> Double {
        let n = indicators.count
        guard chains > 0, n > chains else { return 0 }
        let steps = n / chains
        let p = Double(indicators.filter { $0 }.count) / Double(n)
        guard p > 0, p < 1 else { return 0 }
        let r0 = p * (1 - p)
        var gamma = 0.0
        for lag in 1..<steps {
            var s = 0.0
            for c in 0..<chains {
                for i in 0..<(steps - lag) where indicators[i * chains + c] && indicators[(i + lag) * chains + c] {
                    s += 1
                }
            }
            let rk = s / Double(n - lag * chains) - p * p
            gamma += 2 * (1 - Double(lag) / Double(steps)) * rk / r0
        }
        return max(gamma, 0)
    }
}
