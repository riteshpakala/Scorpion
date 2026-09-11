//
//  Flux2LoRAMapper.swift
//  ScorpionFlux2
//
//  Adapter module names (as ScorpionKit's format-level extraction found them) → FluxKit's
//  linear prefixes. FluxKit uses mflux names, which match diffusers' Flux2Transformer2DModel
//  except `attn.to_out.0` → `attn.to_out` and `time_guidance_embed.timestep_embedder.*` →
//  `time_guidance_embed.*` (checked against both checkpoints' tensor names). Also handled:
//  PEFT/kohya prefixes, kohya's underscore flattening, and BFL's native block names
//  (fused qkv split row-wise with the down projection shared; the final adaLN's
//  shift/scale halves swapped to diffusers' scale/shift order).
//
//  Nothing is guessed silently: every module must land on a linear of the right shape, or it
//  is reported — and an incomplete mapping makes the test inconclusive.
//

import Foundation
import MLX
import ScorpionKit

public struct Flux2LoRAMapping {
    /// FluxKit linear prefix → ΔW.
    public var updates: [String: WeightUpdate]
    /// Adapter modules that could not be placed on a transformer linear.
    public var unmapped: [String]
    /// Modules for parts the executor doesn't run (e.g. the text encoder).
    public var unsupported: [String]
    /// Modules that needed a non-diffusers name mapping (BFL native).
    public var translated: Int
}

public enum Flux2LoRAMapper {
    static let prefixes = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
                           "lora_unet_", "lora_transformer_", "lycoris_", "lora_"]
    static let textEncoderMarkers = ["lora_te", "text_encoder", "te1_", "te2_", "qwen", "text_model"]

    /// - Parameter linearKeys: every linear prefix the transformer loads (mflux names).
    public static func map(_ updates: [String: WeightUpdate], linearKeys: Set<String>) -> Flux2LoRAMapping {
        let underscored = Dictionary(linearKeys.map { ($0.replacingOccurrences(of: ".", with: "_"), $0) }) { a, _ in a }
        var out = Flux2LoRAMapping(updates: [:], unmapped: [], unsupported: [], translated: 0)
        for (name, update) in updates.sorted(by: { $0.key < $1.key }) {
            let lower = name.lowercased()
            if textEncoderMarkers.contains(where: { lower.hasPrefix($0) || lower.contains(".\($0)") }) {
                out.unsupported.append(name)
                continue
            }
            var key = name
            for p in prefixes where key.hasPrefix(p) { key.removeFirst(p.count) }
            key = diffusers(key)
            if linearKeys.contains(key) {
                out.updates[key] = update
                continue
            }
            if let dotted = underscored[key.replacingOccurrences(of: ".", with: "_")] {
                out.updates[dotted] = update
                continue
            }
            if let parts = bfl(key, update), parts.allSatisfy({ linearKeys.contains($0.0) }) {
                for (k, u) in parts { out.updates[k] = u }
                out.translated += 1
                continue
            }
            out.unmapped.append(name)
        }
        return out
    }

    static func diffusers(_ key: String) -> String {
        key.replacingOccurrences(of: ".attn.to_out.0", with: ".attn.to_out")
            .replacingOccurrences(of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
    }

    /// BFL-native module names → (FluxKit prefix, ΔW) parts.
    static func bfl(_ key: String, _ update: WeightUpdate) -> [(String, WeightUpdate)]? {
        func match(_ pattern: String) -> [String]? {
            guard let re = try? NSRegularExpression(pattern: "^" + pattern + "$"),
                  let m = re.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)) else { return nil }
            return (1..<m.numberOfRanges).compactMap { Range(m.range(at: $0), in: key).map { String(key[$0]) } }
        }
        if let g = match(#"double_blocks\.(\d+)\.img_attn\.qkv"#) {
            return zip(["to_q", "to_k", "to_v"], split(update, 3)).map { ("transformer_blocks.\(g[0]).attn.\($0)", $1) }
        }
        if let g = match(#"double_blocks\.(\d+)\.txt_attn\.qkv"#) {
            return zip(["add_q_proj", "add_k_proj", "add_v_proj"], split(update, 3)).map { ("transformer_blocks.\(g[0]).attn.\($0)", $1) }
        }
        let simple: [(String, (String) -> String)] = [
            (#"double_blocks\.(\d+)\.img_attn\.proj"#, { "transformer_blocks.\($0).attn.to_out" }),
            (#"double_blocks\.(\d+)\.txt_attn\.proj"#, { "transformer_blocks.\($0).attn.to_add_out" }),
            (#"double_blocks\.(\d+)\.img_mlp\.0"#, { "transformer_blocks.\($0).ff.linear_in" }),
            (#"double_blocks\.(\d+)\.img_mlp\.2"#, { "transformer_blocks.\($0).ff.linear_out" }),
            (#"double_blocks\.(\d+)\.txt_mlp\.0"#, { "transformer_blocks.\($0).ff_context.linear_in" }),
            (#"double_blocks\.(\d+)\.txt_mlp\.2"#, { "transformer_blocks.\($0).ff_context.linear_out" }),
            (#"single_blocks\.(\d+)\.linear1"#, { "single_transformer_blocks.\($0).attn.to_qkv_mlp_proj" }),
            (#"single_blocks\.(\d+)\.linear2"#, { "single_transformer_blocks.\($0).attn.to_out" }),
        ]
        for (pattern, target) in simple {
            if let g = match(pattern) { return [(target(g[0]), update)] }
        }
        let fixed: [String: String] = [
            "img_in": "x_embedder", "txt_in": "context_embedder",
            "time_in.in_layer": "time_guidance_embed.linear_1", "time_in.out_layer": "time_guidance_embed.linear_2",
            "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
            "single_stream_modulation.lin": "single_stream_modulation.linear",
            "final_layer.linear": "proj_out",
        ]
        if let t = fixed[key] { return [(t, update)] }
        if key == "final_layer.adaLN_modulation.1" { return [("norm_out.linear", swapHalves(update))] }
        return nil
    }

    /// Split ΔW row-wise into `n` equal parts (a fused projection); low rank keeps `down` shared.
    static func split(_ update: WeightUpdate, _ n: Int) -> [WeightUpdate] {
        switch update {
        case .lowRank(let up, let down, let scale):
            let rows = up.dim(0) / n
            return (0..<n).map { .lowRank(up: up[($0 * rows) ..< (($0 + 1) * rows)], down: down, scale: scale) }
        case .dense(let w):
            let rows = w.dim(0) / n
            return (0..<n).map { .dense(w[($0 * rows) ..< (($0 + 1) * rows)]) }
        }
    }

    /// BFL's final modulation emits (shift, scale); diffusers/mflux expect (scale, shift).
    static func swapHalves(_ update: WeightUpdate) -> WeightUpdate {
        func swap(_ a: MLXArray) -> MLXArray {
            let h = a.dim(0) / 2
            return concatenated([a[h ..< (2 * h)], a[0 ..< h]], axis: 0)
        }
        switch update {
        case .lowRank(let up, let down, let scale): return .lowRank(up: swap(up), down: down, scale: scale)
        case .dense(let w): return .dense(swap(w))
        }
    }
}
