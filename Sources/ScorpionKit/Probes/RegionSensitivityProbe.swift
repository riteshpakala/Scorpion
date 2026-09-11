//
//  RegionSensitivityProbe.swift
//  ScorpionKit
//
//  How much of a region's outcome is decided by the noise *inside* its footprint? Noise
//  is spatial, and blocks of initial noise tend to denoise into specific content that
//  travels with them (Lottery Ticket Hypothesis in Denoising, ECCV 2024) — but attention
//  mixes globally. The noise→region Jacobian ∂(M ⊙ x̂₀)/∂ε measures it exactly:
//  concentration ≫ 1 means the face is decided locally by its own seed patch; ≈ 1 means
//  the whole seed contributes.
//

import Foundation
import MLX

public struct RegionSensitivityResult: Codable, Sendable {
    public let region: String
    public let samples: Int
    /// Fraction of |∂f/∂ε| mass inside the region.
    public let insideMassFraction: Double
    /// Fraction of latent elements inside the region.
    public let areaFraction: Double
    /// (inside mass / inside area) / (outside mass / outside area); nil when no mass leaks out.
    public let concentration: Double?
}

public struct RegionSensitivityProbe {
    public var samples = 8

    public init() {}

    public func run(backend: DiffusionBackend, reference: ReferenceImage, crop: CropSpec,
                    region: ProbeRegion, plan: SeedPlan) throws -> RegionSensitivityResult {
        let shape = backend.latentShape(for: reference, crop: crop)
        let x0 = try backend.encode(reference, crop: crop).expandedDimensions(axis: 0)
        let mask = region.weights(crop: crop, latentShape: shape).expandedDimensions(axis: 0)
        let inside = mask .> 0.5

        var massIn = 0.0, massOut = 0.0
        let used = Array(plan.permutations.prefix(samples))
        for p in used {
            let lambda = MLXArray([p.logSNR])
            let (a, s) = backend.alphaSigma(logSNR: lambda)
            let alpha = a.item(Float.self), sigma = s.item(Float.self)
            let cond = try backend.condition(prompt: p.prompt)
            let eps = SeedPlan.noise(p, shape: shape).expandedDimensions(axis: 0)
            let f: (MLXArray) -> MLXArray = { e in
                let xt = alpha * x0 + sigma * e
                let x0hat = (xt - sigma * backend.predictEpsilon(xt, logSNR: lambda, conditions: [cond])) / alpha
                return (mask * x0hat).sum()
            }
            let g = abs(grad(f)(eps))
            massIn += Double(which(inside, g, MLXArray(Float(0))).sum().item(Float.self))
            massOut += Double(which(inside, MLXArray(Float(0)), g).sum().item(Float.self))
        }
        let total = Double(mask.size)
        let inCount = Double(inside.asType(.float32).sum().item(Float.self))
        let area = inCount / total
        let fraction = massIn + massOut > 0 ? massIn / (massIn + massOut) : 0
        let concentration = massOut > 0 && inCount > 0 && inCount < total
            ? (massIn / inCount) / (massOut / (total - inCount)) : nil
        return RegionSensitivityResult(region: region.label, samples: used.count,
                                       insideMassFraction: fraction, areaFraction: area,
                                       concentration: concentration)
    }
}
