//
//  DeterministicSampler.swift
//  ScorpionKit
//
//  Seed → latent for any Denoiser, in log-SNR λ: DDIM (η = 0) or DPM-Solver++(2M)
//  (Lu et al. 2022; λ_dpm = λ/2), with classifier-free guidance ε̂ = ε̂_∅ + w(ε̂_c − ε̂_∅)
//  and an early exit that returns x̂₀ at λ_stop. Matching DDIM inversion maps latents back
//  to seeds on the same grid.
//
//  Decode-free by construction: this produces *latents* only. Nothing in Scorpion decodes
//  them to pixels or returns them to callers outside probe code — probes read statistics.
//

import Foundation
import MLX

public struct DeterministicSampler: Codable, Sendable {
    public enum Solver: String, Codable, Sendable {
        case ddim
        case dpmSolver2M
    }

    public var solver: Solver = .dpmSolver2M
    public var steps = 16
    public var lambdaMin: Float = -10
    public var lambdaMax: Float = 10

    public init(solver: Solver = .dpmSolver2M, steps: Int = 16) {
        self.solver = solver
        self.steps = steps
    }

    /// λ grid from noisy to clean (inclusive).
    public var grid: [Float] {
        let n = max(steps, 2)
        return (0..<n).map { lambdaMin + (lambdaMax - lambdaMin) * Float($0) / Float(n - 1) }
    }

    func scales(_ backend: Denoiser, _ lambda: Float) -> (Float, Float) {
        let (a, s) = backend.alphaSigma(logSNR: MLXArray([lambda]))
        return (a.item(Float.self), s.item(Float.self))
    }

    /// Guided ε̂. `conditions` has one entry per row (or one shared); `guidance == 1`
    /// skips the unconditional pass.
    func epsilon(_ backend: Denoiser, _ x: MLXArray, lambda: Float, conditions: [Conditioning],
                 unconditional: Conditioning?, guidance: Float) -> MLXArray {
        let b = x.dim(0)
        let l = MLXArray([Float](repeating: lambda, count: b))
        guard guidance != 1, let unconditional else {
            return backend.predictEpsilon(x, logSNR: l, conditions: conditions)
        }
        let conds = conditions.count == 1 ? Array(repeating: conditions[0], count: b) : conditions
        let both = backend.predictEpsilon(concatenated([x, x], axis: 0),
                                          logSNR: concatenated([l, l], axis: 0),
                                          conditions: conds + Array(repeating: unconditional, count: b))
        let c = both[0..<b], u = both[b..<(2 * b)]
        return u + guidance * (c - u)
    }

    /// Seeds z (B, …latent), standard normal → x̂₀ (B, …latent).
    /// `stopLogSNR`: return the x̂₀ prediction at the first grid point ≥ it.
    /// `evaluateEachStep: false` keeps the whole graph lazy (for autodiff through the sampler).
    public func sample(_ backend: Denoiser, noise z: MLXArray, conditions: [Conditioning],
                       unconditional: Conditioning? = nil, guidance: Float = 1, stopLogSNR: Float? = nil,
                       evaluateEachStep: Bool = true) -> MLXArray {
        let g = grid
        var x = z * scales(backend, g[0]).1
        var previousX0: MLXArray?
        var previousH: Float?
        for i in 0..<g.count {
            let (a, s) = scales(backend, g[i])
            let eps = epsilon(backend, x, lambda: g[i], conditions: conditions, unconditional: unconditional, guidance: guidance)
            let x0 = (x - s * eps) / a
            if i == g.count - 1 || (stopLogSNR.map { g[i] >= $0 } ?? false) { return x0 }
            let (an, sn) = scales(backend, g[i + 1])
            switch solver {
            case .ddim:
                x = an * x0 + sn * eps
            case .dpmSolver2M:
                let h = (g[i + 1] - g[i]) / 2
                var dPred = x0
                if let px0 = previousX0, let ph = previousH {
                    let r = ph / h
                    dPred = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * px0
                }
                x = (sn / s) * x - an * (exp(-h) - 1) * dPred
                previousX0 = x0
                previousH = h
            }
            if evaluateEachStep { eval(x) }
        }
        return x
    }

    /// DDIM inversion on this sampler's grid: x₀ (B, …latent) → seeds z.
    public func invert(_ backend: Denoiser, x0: MLXArray, conditions: [Conditioning],
                       unconditional: Conditioning? = nil, guidance: Float = 1, steps inversionSteps: Int? = nil) -> MLXArray {
        var fine = self
        if let inversionSteps { fine.steps = inversionSteps }
        let g = fine.grid
        var x = x0 * scales(backend, g[g.count - 1]).0
        for i in stride(from: g.count - 1, to: 0, by: -1) {
            let (a, s) = scales(backend, g[i])
            let eps = epsilon(backend, x, lambda: g[i], conditions: conditions, unconditional: unconditional, guidance: guidance)
            let x0hat = (x - s * eps) / a
            let (ap, sp) = scales(backend, g[i - 1])
            x = ap * x0hat + sp * eps
            eval(x)
        }
        return x / scales(backend, g[0]).1
    }
}
