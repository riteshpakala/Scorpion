//
//  Curvature.swift
//  ScorpionKit
//
//  Coordinate-wise curvature of a diffusion model's noised density — the quantity Kim & Lee
//  (arXiv 2605.26756) use to localize memorization. Second-order Tweedie ties it to how much
//  the model still lets each coordinate vary:
//
//      Cov[x₀ | x_λ] = (σ⁴ ∇² log p_λ + σ² I) / α²,   ∇ log p_λ = −ε̂/σ,   ∇² log p_λ = −J/σ
//      ⇒ Var[x₀,ᵢ | x_λ] = (σ²/α²) · ρᵢ,   ρᵢ = 1 − σ Jᵢᵢ,   J = ∂ε̂/∂x_λ
//
//  ρ is the *retained variance*: 1 = coordinate i is still free at this noise level, 0 = the
//  model has pinned it (a memorized pixel — or natural low dimension, which is why memory
//  maps compare against an underfitted baseline: ρ_base − ρ_target, ρ_uncond − ρ_cond).
//  diag J comes from a `JacobianDiagonalEstimator` (Hutchinson: E[v ⊙ Jᵀv] = E[v ⊙ Jv] =
//  diag J for Rademacher v); the forward-only score difference (ε̂_a − ε̂_b)² is the paper's
//  Fisher-identity surrogate, kept as a labelled quick-look layer.
//

import Foundation
import MLX

public enum Curvature {
    /// ±1 probe vectors from the keyed schedule.
    public static func rademacher(_ schedule: SeedSchedule, _ index: Int, shape: [Int]) -> MLXArray {
        which(schedule.normal(.rademacher, index, shape: shape) .>= 0, MLXArray(Float(1)), MLXArray(Float(-1)))
    }

    /// Exact diag(∂ε̂/∂x) for one point `x` (1, …latent) by one VJP per coordinate, batched.
    /// Small latents only (tests and ground truth).
    public static func exactDiagJacobian(_ backend: Denoiser, x: MLXArray, logSNR: Float,
                                         conditions: [Conditioning]) -> MLXArray {
        let shape = Array(x.shape.dropFirst())
        let d = shape.reduce(1, *)
        let basis = MLXArray.eye(d).reshaped([d] + shape)
        let xs = broadcast(x, to: [d] + shape)
        let l = MLXArray([Float](repeating: logSNR, count: d))
        let f: (MLXArray) -> MLXArray = { xx in (basis * backend.predictEpsilon(xx, logSNR: l, conditions: conditions)).sum() }
        return (basis * grad(f)(xs)).sum(axis: 0, keepDims: true)
    }

    /// ρ = 1 − σ·diag J, with σ per row (B,) broadcast over the latent.
    public static func retainedVariance(diagJacobian j: MLXArray, sigma: MLXArray) -> MLXArray {
        1 - sigma.reshaped([j.dim(0)] + Array(repeating: 1, count: j.ndim - 1)) * j
    }

    /// Forward-only surrogate (ε̂_a − ε̂_b)²: the squared score difference in ε units
    /// (σ²·Δs of the paper), comparable across noise levels.
    public static func scoreDifference(_ a: MLXArray, _ b: MLXArray) -> MLXArray { (a - b).square() }
}

extension AnalyticGMMBackend {
    /// Posterior responsibilities r (B, K) and noised component variances v = α²τₖ² + σ² (B, K).
    private func responsibilities(_ blk: Block, _ xb: MLXArray, alpha: MLXArray, sigma: MLXArray,
                                  conditions: [Conditioning]) -> (r: MLXArray, v: MLXArray) {
        let b = xb.dim(0)
        let dist = xb.square().sum(axis: 1, keepDims: true) - 2 * alpha * matmul(xb, blk.means.transposed())
            + alpha.square() * blk.means.square().sum(axis: 1).reshaped(1, -1)
        let v = alpha.square() * blk.variances.reshaped(1, -1) + sigma.square()
        let logits = blk.rowLogWeights(conditions, batch: b) - 0.5 * Float(blk.dim) * log(v) - dist / (2 * v)
        return (softmax(logits, axis: 1), v)
    }

    /// Exact diag ∇²ₓ log p_λ(x) (B, …latent):
    /// Hᵢᵢ = Σₖ rₖ(gₖᵢ − ḡᵢ)² − Σₖ rₖ/vₖ, gₖ = −(x − α mₖ)/vₖ, ḡ = Σₖ rₖ gₖ
    /// (centered, so Float32 doesn't cancel two large terms).
    public func exactDiagHessian(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning] = []) -> MLXArray {
        let b = x.dim(0)
        let xf = x.reshaped(b, dimension)
        let (a, s) = alphaSigma(logSNR: logSNR)
        let alpha = a.reshaped(b, 1), sigma = s.reshaped(b, 1)
        var out = MLXArray.zeros([b, dimension])
        for blk in blocks {
            let xb = xf * blk.indicator
            let (r, v) = responsibilities(blk, xb, alpha: alpha, sigma: sigma, conditions: conditions)
            let vv = v.expandedDimensions(axis: 2)
            let g = -(xb.expandedDimensions(axis: 1) - alpha.expandedDimensions(axis: 2) * blk.means.expandedDimensions(axis: 0)) / vv
            let rr = r.expandedDimensions(axis: 2)
            let gbar = (rr * g).sum(axis: 1, keepDims: true)
            let h = (rr * (g - gbar).square()).sum(axis: 1) - (r / v).sum(axis: 1, keepDims: true)
            out = out + h * blk.indicator
        }
        return out.reshaped(x.shape)
    }

    /// Exact posterior mean and diagonal variance of x₀ given x_λ (B, …latent), from the
    /// mixture posterior — independent of the Tweedie identity, so it can check it.
    public func exactPosterior(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning] = [])
        -> (mean: MLXArray, variance: MLXArray) {
        let b = x.dim(0)
        let xf = x.reshaped(b, dimension)
        let (a, s) = alphaSigma(logSNR: logSNR)
        let alpha = a.reshaped(b, 1), sigma = s.reshaped(b, 1)
        var mean = MLXArray.zeros([b, dimension]), variance = MLXArray.zeros([b, dimension])
        for blk in blocks {
            let xb = xf * blk.indicator
            let (r, v) = responsibilities(blk, xb, alpha: alpha, sigma: sigma, conditions: conditions)
            let vv = v.expandedDimensions(axis: 2)                              // (B, K, 1)
            let al = alpha.expandedDimensions(axis: 2)                          // (B, 1, 1)
            let tau2 = blk.variances.reshaped(1, -1, 1)                         // (1, K, 1)
            let m = blk.means.expandedDimensions(axis: 0)                       // (1, K, d)
            let pm = m + al * tau2 / vv * (xb.expandedDimensions(axis: 1) - al * m)
            let pv = tau2 * sigma.expandedDimensions(axis: 2).square() / vv
            let rr = r.expandedDimensions(axis: 2)
            let first = (rr * pm).sum(axis: 1, keepDims: true)
            mean = mean + first.squeezed(axis: 1) * blk.indicator
            // Law of total variance, centered: within-component + between-component spread.
            variance = variance + ((rr * pv).sum(axis: 1) + (rr * (pm - first).square()).sum(axis: 1)) * blk.indicator
        }
        return (mean.reshaped(x.shape), variance.reshaped(x.shape))
    }

    /// Exact retained variance ρ = Var[x₀|x_λ]·α²/σ² (B, …latent).
    public func exactRetainedVariance(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning] = []) -> MLXArray {
        let b = x.dim(0)
        let (a, s) = alphaSigma(logSNR: logSNR)
        let lead = [b] + Array(repeating: 1, count: x.ndim - 1)
        return exactPosterior(x, logSNR: logSNR, conditions: conditions).variance
            * (a.square() / s.square()).reshaped(lead)
    }
}
