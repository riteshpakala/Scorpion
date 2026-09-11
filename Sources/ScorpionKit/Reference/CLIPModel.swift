//
//  CLIPModel.swift
//  ScorpionKit
//
//  CLIP (text + vision towers) in plain MLX ops, driven by the HF `config.json`, loading
//  HF-format `model.safetensors`. Used only to *describe the reference* — it is not the
//  target model, so Scorpion stays target-model-agnostic. Default checkpoint is LAION
//  ViT-B/32 (605 MB, safetensors); `openai/clip-vit-large-patch14` also works.
//  Pickle checkpoints are refused by design.
//

import CoreGraphics
import Foundation
import Hub
import MLX

public enum CLIPError: Error, LocalizedError {
    case badTokenizer(String)
    case missingWeight(String)
    case badConfig(String)

    public var errorDescription: String? {
        switch self {
        case .badTokenizer(let s): return "CLIP tokenizer file unreadable: \(s)"
        case .missingWeight(let s): return "CLIP weight missing: \(s)"
        case .badConfig(let s): return "CLIP config unreadable: \(s)"
        }
    }
}

struct CLIPConfig {
    struct Tower {
        let hidden: Int
        let heads: Int
        let layers: Int
        let activation: String
        let eps: Float
    }

    let text: Tower
    let vision: Tower
    let imageSize: Int
    let patchSize: Int

    static func load(_ url: URL) throws -> CLIPConfig {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw CLIPError.badConfig(url.lastPathComponent)
        }
        let t = json["text_config"] as? [String: Any] ?? json["text_config_dict"] as? [String: Any] ?? [:]
        let v = json["vision_config"] as? [String: Any] ?? json["vision_config_dict"] as? [String: Any] ?? [:]
        func tower(_ d: [String: Any], hidden: Int, heads: Int) -> Tower {
            Tower(hidden: d["hidden_size"] as? Int ?? hidden,
                  heads: d["num_attention_heads"] as? Int ?? heads,
                  layers: d["num_hidden_layers"] as? Int ?? 12,
                  activation: d["hidden_act"] as? String ?? "quick_gelu",
                  eps: (d["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5)
        }
        return CLIPConfig(text: tower(t, hidden: 512, heads: 8), vision: tower(v, hidden: 768, heads: 12),
                          imageSize: v["image_size"] as? Int ?? 224, patchSize: v["patch_size"] as? Int ?? 32)
    }
}

public final class CLIPModel: @unchecked Sendable {
    public static let defaultRepo = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    public let repo: String
    public let tokenizer: CLIPTokenizer
    public let logitScale: Float
    let config: CLIPConfig
    let w: [String: MLXArray]

    init(repo: String, directory: URL) throws {
        self.repo = repo
        config = try CLIPConfig.load(directory.appendingPathComponent("config.json"))
        tokenizer = try CLIPTokenizer(directory: directory)
        w = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        logitScale = exp(w["logit_scale"]?.item(Float.self) ?? log(100))
        // Validate up front so a non-HF-CLIP checkpoint fails with an error, not a crash mid-forward.
        let layer = ["layer_norm1.weight", "layer_norm2.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                     "self_attn.v_proj.weight", "self_attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
        var required = [
            "text_model.embeddings.token_embedding.weight", "text_model.embeddings.position_embedding.weight",
            "text_model.final_layer_norm.weight", "text_projection.weight",
            "vision_model.embeddings.class_embedding", "vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.position_embedding.weight", "vision_model.pre_layrnorm.weight",
            "vision_model.post_layernorm.weight", "visual_projection.weight",
        ]
        for i in 0..<config.text.layers { required += layer.map { "text_model.encoder.layers.\(i).\($0)" } }
        for i in 0..<config.vision.layers { required += layer.map { "vision_model.encoder.layers.\(i).\($0)" } }
        if let missing = required.first(where: { w[$0] == nil }) { throw CLIPError.missingWeight(missing) }
    }

    /// Download (once, into Scorpion's cache) and load.
    public static func load(repo: String = defaultRepo, cacheDirectory: URL = FetchCache.defaultDirectory,
                            progress: ((Double) -> Void)? = nil) async throws -> CLIPModel {
        let hub = HubApi(downloadBase: cacheDirectory.appendingPathComponent("hub", isDirectory: true))
        let dir = try await hub.snapshot(from: repo, matching: ["model.safetensors", "config.json", "vocab.json", "merges.txt"]) {
            progress?($0.fractionCompleted)
        }
        return try CLIPModel(repo: repo, directory: dir)
    }

    // MARK: - Encoders

    /// L2-normalized text embeddings, (N, projection).
    public func encodeText(_ texts: [String], batchSize: Int = 64) -> MLXArray {
        var chunks: [MLXArray] = []
        var start = 0
        while start < texts.count {
            let batch = Array(texts[start..<min(start + batchSize, texts.count)])
            chunks.append(encodeTextBatch(batch))
            start += batchSize
        }
        guard !chunks.isEmpty else { return MLXArray.zeros([0, 1]) }
        let out = concatenated(chunks, axis: 0)
        eval(out)
        return out
    }

    func encodeTextBatch(_ texts: [String]) -> MLXArray {
        let ids = texts.map(tokenizer.encode)
        let L = ids.map(\.count).max() ?? 2
        let B = ids.count
        var flat = [Int32](repeating: Int32(tokenizer.endToken), count: B * L)
        var eos = [Float](repeating: 0, count: B * L)
        for (b, row) in ids.enumerated() {
            for (i, t) in row.enumerated() { flat[b * L + i] = Int32(t) }
            eos[b * L + row.count - 1] = 1
        }
        let tokens = MLXArray(flat, [B, L])
        var x = take(weight("text_model.embeddings.token_embedding.weight"), tokens, axis: 0)
        x = x + weight("text_model.embeddings.position_embedding.weight")[0..<L]
        var maskValues = [Float](repeating: 0, count: L * L)
        for i in 0..<L { for j in (i + 1)..<max(i + 1, L) { maskValues[i * L + j] = -1e9 } }
        let mask = MLXArray(maskValues, [L, L])
        for layer in 0..<config.text.layers {
            x = encoderLayer(x, "text_model.encoder.layers.\(layer)", tower: config.text, mask: mask)
        }
        x = layerNorm(x, "text_model.final_layer_norm", eps: config.text.eps)
        let pooled = (x * MLXArray(eos, [B, L, 1])).sum(axis: 1)
        return normalize(matmul(pooled, weight("text_projection.weight").transposed()))
    }

    /// L2-normalized image embeddings for square crops of the reference, (N, projection).
    public func encodeImage(_ reference: ReferenceImage, crops: [CGRect]) -> MLXArray {
        let s = config.imageSize
        var planes: [Float] = []
        for rect in crops {
            let rgb = reference.rgbPlanar(rect: rect, width: s, height: s)
            for c in 0..<3 {
                for i in 0..<(s * s) { planes.append((rgb[c * s * s + i] - Self.mean[c]) / Self.std[c]) }
            }
        }
        let pixels = MLXArray(planes, [crops.count, 3, s, s])
        let out = encodePixels(pixels)
        eval(out)
        return out
    }

    func encodePixels(_ pixels: MLXArray) -> MLXArray {
        let B = pixels.dim(0), P = config.patchSize, g = config.imageSize / P, D = config.vision.hidden
        let patches = pixels.reshaped(B, 3, g, P, g, P).transposed(0, 2, 4, 1, 3, 5).reshaped(B, g * g, 3 * P * P)
        let pw = weight("vision_model.embeddings.patch_embedding.weight").reshaped(D, 3 * P * P)
        var x = matmul(patches, pw.transposed())
        let cls = broadcast(weight("vision_model.embeddings.class_embedding").reshaped(1, 1, D), to: [B, 1, D])
        x = concatenated([cls, x], axis: 1) + weight("vision_model.embeddings.position_embedding.weight")
        x = layerNorm(x, "vision_model.pre_layrnorm", eps: config.vision.eps)
        for layer in 0..<config.vision.layers {
            x = encoderLayer(x, "vision_model.encoder.layers.\(layer)", tower: config.vision, mask: nil)
        }
        let pooled = layerNorm(x[0..., 0, 0...], "vision_model.post_layernorm", eps: config.vision.eps)
        return normalize(matmul(pooled, weight("visual_projection.weight").transposed()))
    }

    // MARK: - Blocks

    func weight(_ name: String) -> MLXArray {
        guard let a = w[name] else { fatalError("CLIP weight missing: \(name)") }
        return a
    }

    func linear(_ x: MLXArray, _ prefix: String) -> MLXArray {
        var y = matmul(x, weight("\(prefix).weight").transposed())
        if let b = w["\(prefix).bias"] { y = y + b }
        return y
    }

    func layerNorm(_ x: MLXArray, _ prefix: String, eps: Float) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) * rsqrt(variance + eps) * weight("\(prefix).weight") + weight("\(prefix).bias")
    }

    func activation(_ x: MLXArray, _ kind: String) -> MLXArray {
        kind == "quick_gelu" ? x * sigmoid(1.702 * x) : 0.5 * x * (1 + erf(x / Float(2).squareRoot()))
    }

    func encoderLayer(_ x: MLXArray, _ p: String, tower: CLIPConfig.Tower, mask: MLXArray?) -> MLXArray {
        var h = x + attention(layerNorm(x, "\(p).layer_norm1", eps: tower.eps), "\(p).self_attn", heads: tower.heads, mask: mask)
        let m = layerNorm(h, "\(p).layer_norm2", eps: tower.eps)
        h = h + linear(activation(linear(m, "\(p).mlp.fc1"), tower.activation), "\(p).mlp.fc2")
        return h
    }

    func attention(_ x: MLXArray, _ p: String, heads: Int, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), D = x.dim(2), hd = D / heads
        let q = (linear(x, "\(p).q_proj") * (1 / Float(hd).squareRoot())).reshaped(B, L, heads, hd).transposed(0, 2, 1, 3)
        let k = linear(x, "\(p).k_proj").reshaped(B, L, heads, hd).transposed(0, 2, 1, 3)
        let v = linear(x, "\(p).v_proj").reshaped(B, L, heads, hd).transposed(0, 2, 1, 3)
        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        if let mask { scores = scores + mask }
        let o = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped(B, L, D)
        return linear(o, "\(p).out_proj")
    }

    func normalize(_ x: MLXArray) -> MLXArray {
        x / maximum(sqrt(x.square().sum(axis: -1, keepDims: true)), MLXArray(Float(1e-8)))
    }
}
