//
//  SeedEndpoint.swift
//  ScorpionKit
//
//  The seed-basin test's building blocks: composing a seed from a moved segment and a fresh
//  surround, judging whether a sampled latent is "the reference" inside the region, and how
//  Gaussian an inverted seed looks there.
//

import Foundation
import MLX

/// Decides whether sampled latents (B, …latent) are the reference: a hit is score ≥ threshold.
public protocol SeedEndpointCriterion {
    var threshold: Double { get }
    func scores(_ x0: MLXArray) -> [Double]
}

/// Negative masked mean squared latent distance to the reference, inside the region.
public struct PhotoCriterion: SeedEndpointCriterion {
    /// (…latent)
    public let reference: MLXArray
    /// (…latent) region weights.
    public let mask: MLXArray
    public let threshold: Double

    public init(reference: MLXArray, mask: MLXArray, threshold: Double) {
        self.reference = reference
        self.mask = mask
        self.threshold = threshold
    }

    public func scores(_ x0: MLXArray) -> [Double] {
        let b = x0.dim(0)
        let d = ((x0 - reference.expandedDimensions(axis: 0)).square() * mask.expandedDimensions(axis: 0))
            .reshaped(b, -1).sum(axis: 1) / mask.sum()
        return (-d).asArray(Float.self).map(Double.init)
    }

    /// Median masked squared distance between distinct images — the scale "the same image"
    /// is judged against. Nil with fewer than two samples.
    public static func scale(samples: [MLXArray], mask: MLXArray) -> Double? {
        guard samples.count >= 2 else { return nil }
        let w = mask.reshaped(1, -1)
        let v = stacked(samples).reshaped(samples.count, -1) * w
        let sq = v.square().sum(axis: 1)
        let n = samples.count
        let d = ((sq.reshaped(-1, 1) + sq.reshaped(1, -1) - 2 * matmul(v, v.transposed())) / w.sum()).asArray(Float.self)
        let pairs = (0..<n).flatMap { i in ((i + 1)..<n).map { Double(d[i * n + $0]) } }.sorted()
        return pairs[pairs.count / 2]
    }
}

public enum SeedComposer {
    /// A seed from its region segment and a surround, through a soft footprint `m`:
    /// m·segment + √(1 − m²)·surround keeps every element N(0, 1).
    public static func compose(_ segment: MLXArray, _ surround: MLXArray, footprint m: MLXArray?) -> MLXArray {
        guard let m else { return segment }
        return m * segment + sqrt(1 - m.square()) * surround
    }
}

public enum SeedTypicality {
    /// χ² z-score of `z` (…latent) over elements where `mask` > 0.5: (Σz² − n)/√(2n).
    /// ≈ 0 for a typical Gaussian draw; a mode (z ≈ 0) or unexplained structure is atypical.
    public static func chi2Z(_ z: MLXArray, mask: MLXArray) -> Double {
        let m = (mask .> 0.5).asType(.float32)
        let n = Double(m.sum().item(Float.self))
        guard n > 0 else { return 0 }
        let ss = Double((z.asType(.float32).square() * m).sum().item(Float.self))
        return (ss - n) / (2 * n).squareRoot()
    }
}
