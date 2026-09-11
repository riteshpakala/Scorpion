//
//  DiffusionBackend.swift
//  ScorpionKit
//
//  The one place a model is executed. Everything a probe needs from a diffusion or flow
//  model reduces to: encode the reference into the model's latent space, map log-SNR λ to
//  signal/noise scales, and predict the noise ε̂ for a noised latent. A safetensors file
//  carries no compute graph, so reference-specific likelihood always needs an executor;
//  the probe math written against this protocol is identical for every model family.
//
//  Phase 1 ships `AnalyticGMMBackend` (closed-form ground truth). Real executors
//  (e.g. FLUX.2 Klein via Frigate's FluxKit) conform to this in Phase 2.
//

import CoreGraphics
import Foundation
import MLX

/// Backend-specific conditioning (e.g. text-encoder output). Opaque to probes.
public struct Conditioning {
    public let prompt: String
    public let payload: Any?

    public init(prompt: String, payload: Any? = nil) {
        self.prompt = prompt
        self.payload = payload
    }
}

public protocol DiffusionBackend: AnyObject {
    var identifier: String { get }

    /// Latent shape (no batch dimension) for a crop of the reference, e.g. [C, H, W].
    func latentShape(for reference: ReferenceImage, crop: CropSpec) -> [Int]

    /// The reference crop in latent space, shape `latentShape`.
    func encode(_ reference: ReferenceImage, crop: CropSpec) throws -> MLXArray

    func condition(prompt: String) throws -> Conditioning

    /// Signal and noise scales at log-SNR λ (λ = log α²/σ²), elementwise over `logSNR` (B,).
    func alphaSigma(logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray)

    /// ε̂ for a batch: `x` (B, …latent), `logSNR` (B,), one conditioning per row (or one
    /// shared). Must be built from MLX ops so region-sensitivity probes can differentiate it.
    func predictEpsilon(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning]) -> MLXArray
}

extension DiffusionBackend {
    /// Variance-preserving parameterization: α² = sigmoid(λ), σ² = sigmoid(−λ).
    /// Flow-matching backends override (α = 1 − t, σ = t with λ = 2 log((1 − t)/t)).
    public func alphaSigma(logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray) {
        (sqrt(sigmoid(logSNR)), sqrt(sigmoid(-logSNR)))
    }
}

/// A region to split probe statistics over: the whole latent, or a face mask.
public struct ProbeRegion: Sendable {
    public let label: String
    let mask: RegionMask?
    /// When set, the soft mask is binarized at this threshold.
    let hardThreshold: Float?
    /// Explicit grid override (row-major H×W), used by synthetic scenarios.
    let explicitGrid: [Float]?

    public static let full = ProbeRegion(label: "full", mask: nil, hardThreshold: nil, explicitGrid: nil)

    public static func face(_ mask: RegionMask, hard: Bool = false) -> ProbeRegion {
        ProbeRegion(label: mask.label, mask: mask, hardThreshold: hard ? 0.5 : nil, explicitGrid: nil)
    }

    public static func grid(_ label: String, _ values: [Float]) -> ProbeRegion {
        ProbeRegion(label: label, mask: nil, hardThreshold: nil, explicitGrid: values)
    }

    public var isFull: Bool { mask == nil && explicitGrid == nil }

    /// Weights broadcast to `latentShape` ([C, H, W] or [H, W]).
    public func weights(crop: CropSpec, latentShape: [Int]) -> MLXArray {
        guard latentShape.count >= 2 else { return MLXArray.ones(latentShape) }
        let h = latentShape[latentShape.count - 2], w = latentShape[latentShape.count - 1]
        var values: [Float]
        if let explicitGrid, explicitGrid.count == h * w {
            values = explicitGrid
        } else if let mask {
            values = mask.grid(crop: crop.rect, width: w, height: h)
        } else {
            return MLXArray.ones(latentShape)
        }
        if let t = hardThreshold { values = values.map { $0 > t ? 1 : 0 } }
        let grid = MLXArray(values, [h, w])
        let lead = Array(repeating: 1, count: latentShape.count - 2)
        return broadcast(grid.reshaped(lead + [h, w]), to: latentShape)
    }
}
