//
//  DiffusionBackend.swift
//  ScorpionKit
//
//  The one place a model is executed. A safetensors file carries no compute graph, so
//  measuring what a model memorized always needs an executor; the probe math written
//  against these protocols is identical for every model family.
//
//  Interface segregation: a probe asks a model for three separate things, and each
//  consumer depends only on the one it uses.
//  - `LatentEncoder`: the reference in the model's latent space, with the frame of the
//    reference it covers and the grid its latent positions form (the heatmap's cells).
//  - `PromptConditioner`: prompt → conditioning.
//  - `Denoiser`: log-SNR λ → (α, σ), and ε̂ for a noised latent. Curvature estimators and
//    samplers need only this.
//
//  ScorpionKit ships `AnalyticGMMBackend` (closed-form ground truth). Real executors live
//  in their own targets (first: `ScorpionFlux2`, FLUX.2 Klein) so this library stays
//  model-agnostic.
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

/// Rows × columns of latent positions — the heatmap's cells.
public struct GridShape: Codable, Sendable, Hashable, CustomStringConvertible {
    public let rows: Int
    public let cols: Int

    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
    }

    public var count: Int { rows * cols }
    public var description: String { "\(rows)×\(cols)" }
}

/// The reference in a model's latent space.
public struct EncodedReference {
    /// Latent (no batch dimension), e.g. [C, H, W]; its last two axes are `grid`.
    public let latent: MLXArray
    /// The part of the reference image (pixels, top-left origin) the latent covers.
    public let frame: CGRect
    public let grid: GridShape

    public init(latent: MLXArray, frame: CGRect) {
        self.latent = latent
        self.frame = frame
        let s = latent.shape
        self.grid = GridShape(rows: s[s.count - 2], cols: s[s.count - 1])
    }

    public var shape: [Int] { latent.shape }

    /// The reference pixel rectangle of grid cell (row, col).
    public func cellRect(row: Int, col: Int) -> CGRect {
        let w = frame.width / CGFloat(grid.cols), h = frame.height / CGFloat(grid.rows)
        return CGRect(x: frame.minX + CGFloat(col) * w, y: frame.minY + CGFloat(row) * h, width: w, height: h)
    }

    /// A grid mask (row-major, rows×cols) broadcast to the latent shape.
    public func latentMask(_ grid: [Float]) -> MLXArray {
        precondition(grid.count == self.grid.count, "mask has \(grid.count) cells, grid has \(self.grid.count)")
        let lead = Array(repeating: 1, count: shape.count - 2)
        return broadcast(MLXArray(grid, lead + [self.grid.rows, self.grid.cols]), to: shape)
    }
}

public protocol ModelIdentity: AnyObject {
    var identifier: String { get }
}

public protocol LatentEncoder: ModelIdentity {
    func encode(_ reference: ReferenceImage) throws -> EncodedReference
}

public protocol PromptConditioner: ModelIdentity {
    func condition(prompt: String) throws -> Conditioning
}

public protocol Denoiser: ModelIdentity {
    /// Signal and noise scales at log-SNR λ (λ = log α²/σ²), elementwise over `logSNR` (B,).
    func alphaSigma(logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray)

    /// ε̂ for a batch: `x` (B, …latent), `logSNR` (B,), one conditioning per row (or one
    /// shared). Must be built from MLX ops so curvature estimators can differentiate it.
    func predictEpsilon(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning]) -> MLXArray

    /// Largest batch one call should carry (memory bound). Probes chunk to it.
    var maxBatch: Int { get }
}

extension Denoiser {
    /// Variance-preserving parameterization: α² = sigmoid(λ), σ² = sigmoid(−λ).
    /// Flow-matching backends override (α = 1 − t, σ = t, i.e. σ = sigmoid(−λ/2)).
    public func alphaSigma(logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray) {
        (sqrt(sigmoid(logSNR)), sqrt(sigmoid(-logSNR)))
    }

    public var maxBatch: Int { 1 << 16 }

    /// (α, σ) at one λ, as scalars.
    public func scales(_ lambda: Float) -> (alpha: Float, sigma: Float) {
        let (a, s) = alphaSigma(logSNR: MLXArray([lambda]))
        return (a.item(Float.self), s.item(Float.self))
    }
}

public typealias DiffusionBackend = LatentEncoder & PromptConditioner & Denoiser

/// What a memorization probe compares: the model under test and its underfitted baseline.
public struct ModelPair {
    public let target: DiffusionBackend
    public let base: DiffusionBackend
    /// Provenance recorded in reports (model link, commit, file hashes, variants).
    public var provenance: [String: String]
    /// Problems that make any finding inconclusive (e.g. the adapter was trained on a
    /// different base, or some of its modules could not be applied).
    public var issues: [String]
    /// The target's trigger words, when its metadata names them (a trigger-gated adapter only
    /// opens its memory under its trigger).
    public var triggers: [String]

    public init(target: DiffusionBackend, base: DiffusionBackend, provenance: [String: String] = [:], issues: [String] = [],
                triggers: [String] = []) {
        self.target = target
        self.base = base
        self.provenance = provenance
        self.issues = issues
        self.triggers = triggers
    }
}
