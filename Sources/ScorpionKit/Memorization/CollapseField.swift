//
//  CollapseField.swift
//  ScorpionKit
//
//  Step 2 — where the model pins the reference. At each selected noise level the reference
//  is noised with keyed draws shared by every model (x_λ = α·x₀ + σ·ε), and each model's
//  retained variance is estimated per coordinate (second-order Tweedie; Kim & Lee Prop. 4.1):
//
//      ρ = 1 − σ·diag(∂ε̂/∂x_λ)       1 = still free here, 0 = pinned
//
//  Collapse is what the target pins that its baseline leaves free, averaged over channels
//  so it stays in ρ units for any latent width (the paper sums; same ranking):
//
//      C  = mean_c(ρ_base − ρ_target)             (weights vs. underfitted baseline, Δh_θ̃)
//      C∅ = mean_c(ρ_uncond − ρ_cond), target     (prompt vs. unconditional, Δh_∅)
//
//  Per-draw maps are kept (each draw has its own probe vectors), so later steps get a
//  standard error per position rather than a bare point estimate.
//

import Foundation
import MLX

public struct CollapseField {
    public let levels: [Float]
    /// adjacency[i]: levels i and i+1 are neighbours (the sustained rule pairs only these).
    public let adjacency: [Bool]
    public let grid: GridShape
    public let draws: Int
    /// level → draw → cell: C = mean_c(ρ_base − ρ_target).
    public let collapse: [[[Float]]]
    /// level → draw → cell: C∅ = mean_c(ρ_uncond − ρ_cond) under the target (conditional prompts only).
    public let promptCollapse: [[[Float]]]?
    /// level → cell, means over draws.
    public let retainedBase: [[Float]]
    public let retainedTarget: [[Float]]
    /// level → cell: forward-only ρ̂ of the base (spread of x̂₀ over draws × α/σ).
    public let spreadBase: [[Float]]
    /// level → cell: mean_c (ε̂_target − ε̂_base)² — the forward-only surrogate.
    public let scoreDifference: [[Float]]
    public let estimator: String
    public let evaluations: Int
}

public struct CollapseFieldStage {
    public let estimator: JacobianDiagonalEstimator
    public let draws: Int

    public init(estimator: JacobianDiagonalEstimator, draws: Int) {
        self.estimator = estimator
        self.draws = max(draws, 2)
    }

    /// The noised reference at λ for `n` keyed draws — identical for every model.
    public static func noised(_ model: Denoiser, x0: MLXArray, logSNR: Float, draws n: Int, schedule: SeedSchedule,
                              stream: SeedSchedule.Stream = .noise) -> (x: MLXArray, alpha: Float, sigma: Float) {
        let (alpha, sigma) = model.scales(logSNR)
        let eps = schedule.normal(stream, SeedSchedule.levelKey(logSNR), shape: [n] + x0.shape)
        return (alpha * x0.expandedDimensions(axis: 0) + sigma * eps, alpha, sigma)
    }

    public func measure(models: ModelPair, reference: EncodedReference, prompt: ProbePrompt, levels: [Float],
                        adjacency: [Bool], schedule: SeedSchedule) throws -> CollapseField {
        let grid = reference.grid, n = draws
        let condT = try models.target.condition(prompt: prompt.text)
        let condB = try models.base.condition(prompt: prompt.text)
        let uncT = prompt.isConditional ? try models.target.condition(prompt: ProbePrompt.unconditional) : nil
        var collapse: [[[Float]]] = [], prompted: [[[Float]]] = []
        var rhoB: [[Float]] = [], rhoT: [[Float]] = [], spreadB: [[Float]] = [], scoreDiff: [[Float]] = []
        var evaluations = 0

        for lambda in levels {
            let (xt, alpha, sigma) = Self.noised(models.target, x0: reference.latent, logSNR: lambda, draws: n, schedule: schedule)
            // The same probe vectors for every model and prompt: the differences are paired.
            let key = ProbeKey(schedule: schedule, level: SeedSchedule.levelKey(lambda))
            let t = estimator.estimate(models.target, x: xt, logSNR: lambda, conditions: [condT], key: key)
            let b = estimator.estimate(models.base, x: xt, logSNR: lambda, conditions: [condB], key: key)
            let pT = 1 - sigma * t.diagonal, pB = 1 - sigma * b.diagonal
            collapse.append(GridMath.rows(pB - pT, grid: grid))
            rhoB.append(GridMath.mean(GridMath.rows(pB, grid: grid)))
            rhoT.append(GridMath.mean(GridMath.rows(pT, grid: grid)))
            if let uncT {
                let u = estimator.estimate(models.target, x: xt, logSNR: lambda, conditions: [uncT], key: key)
                prompted.append(GridMath.rows((1 - sigma * u.diagonal) - pT, grid: grid))
            }
            evaluations += n * estimator.evaluationsPerRow * (uncT == nil ? 2 : 3)

            let x0B = (xt - sigma * b.epsilon) / alpha
            let spread = sqrt(x0B.variance(axis: 0, keepDims: true, ddof: 1)) * (alpha / sigma)
            spreadB.append(GridMath.rows(spread, grid: grid)[0])
            scoreDiff.append(GridMath.mean(GridMath.rows(Curvature.scoreDifference(t.epsilon, b.epsilon), grid: grid)))
        }
        return CollapseField(levels: levels, adjacency: adjacency, grid: grid, draws: n, collapse: collapse,
                             promptCollapse: uncT == nil ? nil : prompted, retainedBase: rhoB, retainedTarget: rhoT,
                             spreadBase: spreadB, scoreDifference: scoreDiff, estimator: estimator.name,
                             evaluations: evaluations)
    }
}
