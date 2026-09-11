//
//  Flux2KleinBackend.swift
//  ScorpionFlux2
//
//  FLUX.2 Klein as a `DiffusionBackend`. Flow matching: x_t = (1 − σ)·x₀ + σ·ε, so
//  α = 1 − σ and λ = log(α²/σ²) gives σ = sigmoid(−λ/2). The transformer predicts the
//  velocity v ≈ ε − x₀; the probe's ε̂ = x_t + (1 − σ)·v̂ (exact algebra, in float32), and
//  Tweedie's retained variance ρ = 1 − σ·diag(∂ε̂/∂x) holds for any (α, σ).
//
//  Latents are the VAE's BN-normalized latents, unpacked to [32, H/8, W/8] so the heatmap
//  has one cell per 8×8 pixels (64×64 at 512²); the transformer sees them as 2×2-packed
//  128-channel tokens. Target and base are two views of one `Flux2Model` that differ only
//  in the adapter switch.
//

import CoreGraphics
import FluxKit
import Foundation
import MLX
import ScorpionKit

/// The loaded Klein model shared by the target and base backends.
public final class Flux2Model: @unchecked Sendable {
    public let directory: URL
    public let transformer: Flux2Transformer
    public let adapterSwitch: AdapterSwitch
    /// Longest side the reference is resized to (a multiple of 16).
    public let longSide: Int
    public let transformerSource: String
    private let pipeline: Flux2Pipeline
    private var embeddings: [String: MLXArray] = [:]
    private var ids: [String: MLXArray] = [:]
    private let lock = NSRecursiveLock()

    public init(directory: URL, transformer: Flux2Transformer, adapterSwitch: AdapterSwitch, longSide: Int,
                transformerSource: String) {
        self.directory = directory
        self.transformer = transformer
        self.adapterSwitch = adapterSwitch
        self.longSide = max(16, longSide / 16 * 16)
        self.transformerSource = transformerSource
        self.pipeline = Flux2Pipeline(modelDirectory: directory)
    }

    /// Qwen3 prompt embeddings (1, 512, 7680), cached; the text encoder is loaded once per
    /// batch of new prompts and freed afterwards.
    public func embeddings(for prompts: [String]) throws -> [MLXArray] {
        try lock.withLock {
            let missing = Array(Set(prompts.filter { embeddings[$0] == nil }))
            if !missing.isEmpty {
                let encoder = try Qwen3TextEncoder(componentDir: directory.appendingPathComponent("text_encoder"))
                for p in missing {
                    let (idsArray, mask) = try pipeline.tokenize(prompt: p)
                    let e = encoder.promptEmbeddings(inputIDs: idsArray, attentionMask: mask)
                    eval(e)
                    embeddings[p] = e
                }
                Memory.clearCache()
            }
            return prompts.map { embeddings[$0]! }
        }
    }

    func positionIDs(packedH: Int, packedW: Int, text: Int) -> (MLXArray, MLXArray) {
        lock.withLock {
            let key = "\(packedH)x\(packedW)|\(text)"
            if let img = ids[key + "|img"], let txt = ids[key + "|txt"] { return (img, txt) }
            let img = Flux2Pipeline.latentIDs(packedH: packedH, packedW: packedW), txt = Flux2Pipeline.textIDs(count: text)
            ids[key + "|img"] = img
            ids[key + "|txt"] = txt
            return (img, txt)
        }
    }

    /// Pixel size the reference is resized to: aspect preserved, long side `longSide`, multiples of 16.
    public func workingSize(for image: ReferenceImage) -> (width: Int, height: Int) {
        let s = Double(longSide) / Double(max(image.width, image.height))
        func snap(_ v: Double) -> Int { max(16, Int((v / 16).rounded()) * 16) }
        return (snap(Double(image.width) * s), snap(Double(image.height) * s))
    }

    /// The reference in Klein's latent space: (32, H/8, W/8), float32; the frame is the whole image.
    public func encode(_ image: ReferenceImage) throws -> EncodedReference {
        let (w, h) = workingSize(for: image)
        let px = image.rgbPlanar(width: w, height: h)                 // (3, h, w) in [0, 1]
        let nhwc = (MLXArray(px, [3, h, w]).transposed(1, 2, 0) * 2 - 1).expandedDimensions(axis: 0)
        let encoder = try Flux2VAEEncoder(componentDir: directory.appendingPathComponent("vae"))
        let packed = encoder.encodePackedNormalized(nhwc).asType(.float32)   // (1, S, 128)
        let latent = Flux2Packing.unpack(packed, packedH: h / 16, packedW: w / 16).squeezed(axis: 0)
        eval(latent)
        Memory.clearCache()
        return EncodedReference(latent: latent, frame: image.bounds)
    }
}

/// Rectified-flow noise schedule in log-SNR: x_t = (1 − σ)·x₀ + σ·ε, λ = log(α²/σ²).
public enum FlowSchedule {
    public static func alphaSigma(_ logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray) {
        let s = sigmoid(-logSNR / 2)
        return (1 - s, s)
    }

    public static func sigma(_ lambda: Float) -> Float { 1 / (1 + exp(lambda / 2)) }
}

public enum Flux2Packing {
    /// (B, 32, 2Hp, 2Wp) → (B, Hp·Wp, 128); channel = c·4 + dy·2 + dx (FLUX.2's 2×2 patchify).
    public static func pack(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4)
            .reshaped(b, c * 4, (h / 2) * (w / 2)).transposed(0, 2, 1)
    }

    /// (B, Hp·Wp, 128) → (B, 32, 2Hp, 2Wp).
    public static func unpack(_ t: MLXArray, packedH: Int, packedW: Int) -> MLXArray {
        let b = t.dim(0), c = t.dim(2) / 4
        return t.transposed(0, 2, 1).reshaped(b, c, 2, 2, packedH, packedW).transposed(0, 1, 4, 2, 5, 3)
            .reshaped(b, c, 2 * packedH, 2 * packedW)
    }
}

public final class Flux2KleinBackend: DiffusionBackend {
    public let identifier: String
    public let model: Flux2Model
    /// Whether this view runs with the adapter (the target) or without (the base).
    public let adapterOn: Bool
    public var maxBatch: Int

    public init(identifier: String, model: Flux2Model, adapterOn: Bool, maxBatch: Int = 4) {
        self.identifier = identifier
        self.model = model
        self.adapterOn = adapterOn
        self.maxBatch = maxBatch
    }

    /// Flow matching: σ = sigmoid(−λ/2), α = 1 − σ.
    public func alphaSigma(logSNR: MLXArray) -> (alpha: MLXArray, sigma: MLXArray) { FlowSchedule.alphaSigma(logSNR) }

    public func encode(_ reference: ReferenceImage) throws -> EncodedReference { try model.encode(reference) }

    public func condition(prompt: String) throws -> Conditioning {
        Conditioning(prompt: prompt, payload: try model.embeddings(for: [prompt])[0])
    }

    /// All rows share one noise level (the transformer takes a scalar timestep) and one prompt.
    public func predictEpsilon(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning]) -> MLXArray {
        let b = x.dim(0)
        let lambda = logSNR[0].item(Float.self)
        let sigma = FlowSchedule.sigma(lambda)
        guard let embed = conditions.first?.payload as? MLXArray else {
            preconditionFailure("Flux2KleinBackend needs conditioning from its own condition(prompt:)")
        }
        precondition(conditions.allSatisfy { $0.prompt == conditions[0].prompt }, "one prompt per batch")
        model.adapterSwitch.enabled = adapterOn
        let encoder = b == 1 ? embed : broadcast(embed, to: [b] + Array(embed.shape.dropFirst()))
        let (hp, wp) = (x.dim(2) / 2, x.dim(3) / 2)
        let (imgIDs, txtIDs) = model.positionIDs(packedH: hp, packedW: wp, text: embed.dim(1))
        let v = model.transformer(hidden: Flux2Packing.pack(x), encoder: encoder, timestep: sigma * 1000,
                                  imgIDs: imgIDs, txtIDs: txtIDs)
        return x + (1 - sigma) * Flux2Packing.unpack(v.asType(.float32), packedH: hp, packedW: wp)
    }
}
