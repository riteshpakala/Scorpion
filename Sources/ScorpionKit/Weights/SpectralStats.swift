//
//  SpectralStats.swift
//  ScorpionKit
//
//  Model-agnostic spectral analysis of a weight update ΔW — pure linear algebra on 2-D
//  arrays, so it applies to any architecture. This is where "entropy" legitimately lives
//  in the weights: with pᵢ = σᵢ²/Σσ², the spectral entropy H = −Σ pᵢ log pᵢ measures how
//  many directions carry the change. Low entropy ⇒ a few directions dominate ⇒ a narrow
//  learned concept (one identity, one object); high entropy ⇒ a broad shift (style,
//  domain). The top left singular vector u₁ is "what the update injects" — the
//  inference-free fingerprint used for screening (arXiv 2607.25750).
//
//  Low-rank updates are exact via thin QR: up = Qᵤ Rᵤ, downᵀ = Q_d R_d ⇒
//  ΔW = Qᵤ (s·Rᵤ R_dᵀ) Q_dᵀ, so the SVD of ΔW is the SVD of an r×r core.
//

import Foundation
import MLX

public struct SpectralSummary: Codable, Sendable {
    public let module: String
    public let outDim: Int
    public let inDim: Int
    /// Number of singular values the spectrum was computed over (r for LoRA, min(out,in) dense).
    public let spectrumSize: Int
    public let numericRank: Int
    public let topSingularValues: [Float]
    public let frobeniusNorm: Float
    /// Spectral entropy in nats.
    public let spectralEntropy: Float
    /// H / log(spectrumSize) ∈ [0, 1]; 0 = rank-1-like, 1 = isotropic.
    public let normalizedEntropy: Float
    /// exp(H): the "number of directions" carrying the update.
    public let effectiveRank: Float
    /// σ₁² / Σσ².
    public let concentration: Float
    /// ‖ΔW‖ / ‖W_base‖ when a base weight was available.
    public var relativeUpdate: Float?
}

public struct SpectralResult {
    public let summary: SpectralSummary
    /// Sign-canonical top left singular vector (outDim,), float32.
    public let u1: MLXArray
}

public enum SpectralAnalyzer {
    /// Dense updates larger than this on their short side are skipped (cost ~ n³).
    public static let maxDenseDim = 4096
    static let topK = 16

    public static func analyze(module: String, update: WeightUpdate) -> SpectralResult? {
        switch update {
        case .lowRank(let up, let down, let scale):
            let r = down.dim(0)
            if r <= up.dim(0), r <= down.dim(1) {
                return lowRank(module: module, up: up.asType(.float32), down: down.asType(.float32), scale: scale)
            }
            return dense(module: module, w: update.materialized.asType(.float32))
        case .dense(let w):
            return dense(module: module, w: w.asType(.float32))
        }
    }

    static func lowRank(module: String, up: MLXArray, down: MLXArray, scale: Float) -> SpectralResult? {
        let (qu, ru) = MLXLinalg.qr(up, stream: .cpu)                         // (out×r), (r×r)
        let (_, rd) = MLXLinalg.qr(down.transposed(), stream: .cpu)           // (in×r),  (r×r)
        let core = matmul(ru, rd.transposed()) * scale
        let (uc, s, _) = MLXLinalg.svd(core, stream: .cpu)
        let u1 = matmul(qu, uc[0..., 0..<1]).reshaped(-1)
        eval(s, u1)
        return summarize(module: module, outDim: up.dim(0), inDim: down.dim(1),
                         sigmas: s.asArray(Float.self), u1: u1)
    }

    static func dense(module: String, w: MLXArray) -> SpectralResult? {
        let (out, inn) = (w.dim(0), w.dim(1))
        guard min(out, inn) <= maxDenseDim else { return nil }
        // Eigen-decompose the smaller Gram matrix: eigenvalues are σ².
        let u1: MLXArray
        let eigenvalues: MLXArray
        if out <= inn {
            let (vals, vecs) = MLXLinalg.eigh(matmul(w, w.transposed()), stream: .cpu)
            eigenvalues = vals
            u1 = vecs[0..., (out - 1)..<out].reshaped(-1)
        } else {
            let (vals, vecs) = MLXLinalg.eigh(matmul(w.transposed(), w), stream: .cpu)
            eigenvalues = vals
            let v1 = vecs[0..., (inn - 1)..<inn]
            let wv = matmul(w, v1).reshaped(-1)
            u1 = wv / maximum(sqrt((wv * wv).sum()), MLXArray(Float(1e-12)))
        }
        eval(eigenvalues, u1)
        let sigmas = eigenvalues.asArray(Float.self).reversed().map { max($0, 0).squareRoot() }
        return summarize(module: module, outDim: out, inDim: inn, sigmas: sigmas, u1: u1)
    }

    static func summarize(module: String, outDim: Int, inDim: Int, sigmas: [Float], u1: MLXArray) -> SpectralResult? {
        let s = sigmas.sorted(by: >)
        let energy = s.sumOfSquares
        guard energy > 0, energy.isFinite else { return nil }
        var h: Float = 0
        for v in s where v > 0 {
            let p = v * v / energy
            h -= p * log(p)
        }
        let k = s.count
        let normalized = k >= 2 ? h / log(Float(k)) : 0
        let tol = (s.first ?? 0) * 1e-4
        let summary = SpectralSummary(
            module: module, outDim: outDim, inDim: inDim, spectrumSize: k,
            numericRank: s.filter { $0 > tol }.count,
            topSingularValues: Array(s.prefix(topK)),
            frobeniusNorm: energy.squareRoot(),
            spectralEntropy: h, normalizedEntropy: min(max(normalized, 0), 1),
            effectiveRank: exp(h), concentration: (s[0] * s[0]) / energy,
            relativeUpdate: nil)
        return SpectralResult(summary: summary, u1: canonicalSign(u1))
    }

    /// Flip so the largest-magnitude entry is positive (SVD sign is arbitrary).
    static func canonicalSign(_ v: MLXArray) -> MLXArray {
        let flat = v.asType(.float32)
        let idx = argMax(abs(flat)).item(Int.self)
        return flat[idx].item(Float.self) < 0 ? -flat : flat
    }
}
