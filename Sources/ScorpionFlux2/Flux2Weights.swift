//
//  Flux2Weights.swift
//  ScorpionFlux2
//
//  Where the Klein weights come from. FluxKit loads the mflux export (per-component folders
//  of index-sharded safetensors, 4-bit affine); the model-agnostic weight extraction in
//  ScorpionKit handles the adapter under test. This file adds BFL's diffusers transformer —
//  the layout FLUX.2-klein-base-4B ships in — renamed to FluxKit's keys, so the probe can run
//  on the undistilled base that Klein LoRAs are trained on.
//
//  The distilled Klein only ever evaluates four noise levels (σ ≈ 1, .96, .88, .72); the
//  probe needs σ ≤ 0.5, where the base releases the image. Its predictions there are
//  untrained, so runs on the distilled transformer are marked inconclusive (plumbing only).
//

import Foundation
import Hub
import MLX

public enum Flux2Weights {
    public static let baseRepo = "black-forest-labs/FLUX.2-klein-base-4B"
    public static let distilledRepo = "mlx-community/flux2-klein-4b-4bit"

    /// The mflux export with the text encoder, VAE and tokenizer (and, for plumbing runs, the
    /// distilled transformer): `SCORPION_FLUX2_DIR`, else the Swift Hub default location.
    public static var defaultDirectory: URL {
        if let env = ProcessInfo.processInfo.environment["SCORPION_FLUX2_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("huggingface/models/\(distilledRepo)")
    }

    public enum WeightsError: Error, LocalizedError {
        case missing(String)
        case layout(String)

        public var errorDescription: String? {
            switch self {
            case .missing(let s): return "FLUX.2 Klein weights not found: \(s). Download \(distilledRepo) (text encoder, VAE, tokenizer) or set SCORPION_FLUX2_DIR / --option base-dir."
            case .layout(let s): return s
            }
        }
    }

    public static func validate(_ dir: URL, needsTransformer: Bool) throws {
        var components = ["text_encoder", "vae", "tokenizer"]
        if needsTransformer { components.append("transformer") }
        for c in components where !FileManager.default.fileExists(atPath: dir.appendingPathComponent(c).path) {
            throw WeightsError.missing(dir.appendingPathComponent(c).path)
        }
    }

    /// FluxKit (mflux) key for a diffusers Flux2Transformer2DModel key.
    public static func fluxKitKey(_ diffusersKey: String) -> String {
        diffusersKey.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
            .replacingOccurrences(of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
    }

    /// A diffusers transformer (a .safetensors file, or a folder holding one or an index),
    /// renamed to FluxKit's keys. `quantizeBits` re-quantizes 2-D weights (4 or 8); nil keeps bf16.
    public static func diffusersTransformer(at url: URL, quantizeBits: Int? = nil) throws -> [String: MLXArray] {
        var files: [URL] = []
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { throw WeightsError.missing(url.path) }
        if isDir.boolValue {
            files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
        } else {
            files = [url]
        }
        guard !files.isEmpty else { throw WeightsError.missing("*.safetensors in \(url.path)") }
        var out: [String: MLXArray] = [:]
        for f in files {
            for (key, value) in try loadArrays(url: f) {
                let k = fluxKitKey(key)
                if let bits = quantizeBits, k.hasSuffix(".weight"), value.ndim == 2, value.dim(1) % 64 == 0 {
                    let prefix = String(k.dropLast(".weight".count))
                    let q = quantized(value, groupSize: 64, bits: bits)
                    out[prefix + ".weight"] = q.wq
                    out[prefix + ".scales"] = q.scales
                    if let b = q.biases { out[prefix + ".biases"] = b }
                } else {
                    out[k] = value
                }
            }
        }
        guard out["x_embedder.weight"] != nil || out["x_embedder.scales"] != nil else {
            throw WeightsError.layout("\(url.path) is not a FLUX.2 transformer (no x_embedder)")
        }
        return out
    }

    /// FLUX.2-klein-base-4B's diffusers transformer, downloaded once (≈ 7.8 GB).
    public static func baseTransformer(progress: ((Double) -> Void)? = nil) async throws -> URL {
        let dir = try await HubApi().snapshot(from: baseRepo, matching: ["transformer/*"]) { p in
            progress?(p.fractionCompleted)
        }
        return dir.appendingPathComponent("transformer")
    }

    /// "base" or "distilled", from a folder's model card / config (nil when it doesn't say).
    public static func variant(of url: URL) -> String? {
        let candidates = [url, url.deletingLastPathComponent()].flatMap {
            [$0.appendingPathComponent("README.md"), $0.appendingPathComponent("config.json"),
             $0.appendingPathComponent("transformer/config.json")]
        }
        for file in candidates {
            guard let text = try? String(contentsOf: file, encoding: .utf8).lowercased() else { continue }
            if text.contains("klein-base") || text.contains("klein_base") { return "base" }
            if text.contains("distilled") || text.contains("flux.2-klein-4b") || text.contains("flux2-klein-4b") { return "distilled" }
        }
        return nil
    }
}
