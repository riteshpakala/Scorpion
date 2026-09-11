//
//  SeedFootprint.swift
//  ScorpionKit
//
//  Which part of the seed decides the face? The face's pixel footprint is a start, but
//  attention spreads a region's dependence across the whole seed, so the footprint is
//  *measured*: the influence map I(p) = E|∂(M ⊙ x̂₀)/∂ε|ₚ, either one denoising step deep
//  (cheap, any model) or through the whole few-step sampler (exact, small models). The
//  footprint F is the face mask plus the highest-influence seed positions that together
//  carry `coverage` of the influence — the "segment of the seed" the needle lives in.
//  |F|/d is the haystack reduction.
//

import Foundation
import MLX

public struct SeedFootprint: Codable, Sendable {
    public enum Method: String, Codable, Sendable {
        case oneStep = "one-step"
        case throughSampler = "through-sampler"
    }

    public let label: String
    public let shape: [Int]
    /// Soft footprint m ∈ [0, 1] per latent element (row-major).
    public let weights: [Float]
    /// Normalized influence map (sums to 1).
    public let influence: [Float]
    /// Σm / d — the share of the seed the needle search runs over.
    public let fraction: Double
    /// Influence carried by positions with m > 0.5.
    public let capturedInfluence: Double
    public let method: Method

    public var mask: MLXArray { MLXArray(weights, shape) }

    /// A footprint that is exactly a given mask (no influence measurement).
    public static func fixed(_ label: String, weights: [Float], shape: [Int]) -> SeedFootprint {
        let total = Float(weights.count)
        return SeedFootprint(label: label, shape: shape, weights: weights,
                             influence: weights.map { $0 / max(weights.reduce(0, +), 1e-9) },
                             fraction: Double(weights.reduce(0, +) / total), capturedInfluence: 1, method: .oneStep)
    }

    /// IoU of the hard (m > 0.5) footprints.
    public static func iou(_ a: SeedFootprint, _ b: SeedFootprint) -> Double {
        var inter = 0, union = 0
        for (x, y) in zip(a.weights, b.weights) {
            let p = x > 0.5, q = y > 0.5
            if p && q { inter += 1 }
            if p || q { union += 1 }
        }
        return union == 0 ? 1 : Double(inter) / Double(union)
    }
}

public struct SeedFootprintProbe {
    public var samples = 8
    public var band = LogSNRBand.identity
    public var coverage = 0.9
    public var method = SeedFootprint.Method.oneStep
    public var sampler = DeterministicSampler(solver: .ddim, steps: 8)

    public init() {}

    /// `x0`: reference latent (latent shape); `region`: face weights (latent shape).
    public func run(backend: DiffusionBackend, x0: MLXArray, region: MLXArray, label: String,
                    condition: Conditioning, schedule: SeedSchedule) -> SeedFootprint {
        let shape = x0.shape
        let batch = [1] + shape
        let m = region.reshaped(batch)
        var influence = MLXArray.zeros(batch)
        for k in 0..<samples {
            switch method {
            case .oneStep:
                let lambda = band.lo + band.width * (Float(k) + 0.5) / Float(samples)
                let (a, s) = backend.alphaSigma(logSNR: MLXArray([lambda]))
                let alpha = a.item(Float.self), sigma = s.item(Float.self)
                let l = MLXArray([lambda])
                let x = x0.reshaped(batch)
                let f: (MLXArray) -> MLXArray = { e in
                    let xt = alpha * x + sigma * e
                    return (m * (xt - sigma * backend.predictEpsilon(xt, logSNR: l, conditions: [condition])) / alpha).sum()
                }
                influence = influence + abs(grad(f)(schedule.normal(.noise, 50_000 + k, shape: batch)))
            case .throughSampler:
                let f: (MLXArray) -> MLXArray = { z in
                    (m * sampler.sample(backend, noise: z, conditions: [condition], evaluateEachStep: false)).sum()
                }
                influence = influence + abs(grad(f)(schedule.normal(.needle, 60_000 + k, shape: batch)))
            }
            eval(influence)
        }
        let infl = influence.reshaped(-1).asArray(Float.self)
        let regionValues = region.reshaped(-1).asArray(Float.self)
        return Self.footprint(label: label, shape: shape, influence: infl, region: regionValues,
                              coverage: coverage, method: method)
    }

    /// Face mask ∪ the smallest set of highest-influence positions carrying `coverage`.
    static func footprint(label: String, shape: [Int], influence raw: [Float], region: [Float],
                          coverage: Double, method: SeedFootprint.Method) -> SeedFootprint {
        let total = raw.reduce(0, +)
        let influence = total > 0 ? raw.map { $0 / total } : raw
        var weights = region.map { min(max($0, 0), 1) }
        var acc = 0.0
        for i in influence.indices.sorted(by: { influence[$0] > influence[$1] }) {
            if acc >= coverage { break }
            weights[i] = 1
            acc += Double(influence[i])
        }
        let captured = zip(weights, influence).reduce(0.0) { $0 + ($1.0 > 0.5 ? Double($1.1) : 0) }
        return SeedFootprint(label: label, shape: shape, weights: weights, influence: influence,
                             fraction: Double(weights.reduce(0, +)) / Double(max(weights.count, 1)),
                             capturedInfluence: captured, method: method)
    }
}
