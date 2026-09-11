//
//  LevelScout.swift
//  ScorpionKit
//
//  Step 1 — which noise levels to measure. Collapse means something only where the base
//  leaves the reference free (0.5 ≤ ρ_base ≤ 1.1): at lower noise every model pins the
//  image, at higher noise a base that doesn't know the content hesitates between contents.
//  The scout looks at the *base only*, on its own draws, so the level choice can never
//  depend on what the target does.
//

import Foundation
import MLX

public struct LevelScoutResult: Codable, Sendable {
    public struct Level: Codable, Sendable {
        public let logSNR: Float
        /// Share of positions the base releases (smoothed ρ_base within the release range).
        public let releasedFraction: Double
        /// Mean retained variance of the base over the frame.
        public let meanRetained: Double
        public let selected: Bool
    }

    public let levels: [Level]
    public var selected: [Float] { levels.filter(\.selected).map(\.logSNR) }
    /// adjacency[i]: selected levels i and i+1 are neighbours on the scout grid.
    public let adjacency: [Bool]
    public let evaluations: Int
}

public struct LevelScout {
    public let configuration: MemorizationConfiguration

    public init(configuration: MemorizationConfiguration) { self.configuration = configuration }

    public func run(base: DiffusionBackend, reference: EncodedReference, prompt: ProbePrompt,
                    schedule: SeedSchedule) throws -> LevelScoutResult {
        let c = configuration
        let n = max(c.scoutDraws, 1)
        let estimator = c.estimator == .exact ? ExactDiagonal() as JacobianDiagonalEstimator
            : c.estimator.make(probes: c.scoutProbes)
        let cond = try base.condition(prompt: prompt.text)
        var measured: [(Float, Double, Double)] = []
        for lambda in c.scoutLevels {
            let key = SeedSchedule.levelKey(lambda)
            let (alpha, sigma) = base.scales(lambda)
            let eps = schedule.normal(.scout, key, shape: [n] + reference.shape)
            let xt = alpha * reference.latent.expandedDimensions(axis: 0) + sigma * eps
            let d = estimator.estimate(base, x: xt, logSNR: lambda, conditions: [cond],
                                       key: ProbeKey(schedule: schedule, level: key &+ 7))
            let rho = GridMath.mean(GridMath.rows(1 - sigma * d.diagonal, grid: reference.grid))
            let smooth = GridMath.boxSmooth(rho, grid: reference.grid, radius: c.gateSmoothing)
            let released = smooth.filter { $0 >= c.releaseThreshold && $0 <= c.mixingLimit }.count
            measured.append((lambda, Double(released) / Double(max(smooth.count, 1)),
                             Double(rho.reduce(0, +)) / Double(max(rho.count, 1))))
        }

        // The best contiguous window (on the scout grid) of at most `maxLevels` passing levels.
        let passing = measured.map { $0.1 >= c.scoutMinReleased }
        var best: Range<Int>?, bestScore = -1.0
        var i = 0
        while i < measured.count {
            guard passing[i] else { i += 1; continue }
            var j = i
            while j < measured.count && passing[j] { j += 1 }
            let width = min(j - i, max(c.maxLevels, 1))
            for s in i...(j - width) {
                let score = measured[s..<(s + width)].reduce(0) { $0 + $1.1 }
                if score > bestScore { bestScore = score; best = s..<(s + width) }
            }
            i = j
        }
        let levels = measured.enumerated().map { k, m in
            LevelScoutResult.Level(logSNR: m.0, releasedFraction: m.1, meanRetained: m.2, selected: best?.contains(k) ?? false)
        }
        let count = best?.count ?? 0
        return LevelScoutResult(levels: levels, adjacency: [Bool](repeating: true, count: max(count - 1, 0)),
                                evaluations: c.scoutLevels.count * n * estimator.evaluationsPerRow)
    }
}
