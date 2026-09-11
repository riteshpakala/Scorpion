//
//  StructuralHash.swift
//  ScorpionKit
//
//  Architecture identity from structure alone. The family key hashes the set of
//  (index-stripped module pattern, shape) pairs, so two adapters for the same base and
//  target-module set share a key regardless of rank or training data — fingerprints are
//  only compared within a family. Family knowledge stays in data, never in code.
//

import Foundation

public enum StructuralHash {
    /// "lora_unet_down_blocks_1_attentions_0_proj" → "lora_unet_down_blocks_#_attentions_#_proj"
    public static func pattern(_ name: String) -> String {
        name.replacingOccurrences(of: "[0-9]+", with: "#", options: .regularExpression)
    }

    /// Family key over module patterns and dims (rank excluded).
    public static func familyKey(_ modules: [ModuleSpec]) -> String {
        let lines = Set(modules.map { "\(pattern($0.name))|\($0.outDim)x\($0.inDim)" }).sorted()
        return String(Hashing.sha256Hex(lines.joined(separator: "\n")).prefix(16))
    }

    /// Exact structure key over tensor names, shapes and dtypes.
    public static func exactKey(_ tensors: [String: TensorRecord]) -> String {
        let lines = tensors.values.map { "\($0.name)|\($0.shape.map(String.init).joined(separator: "x"))|\($0.rawDType)" }.sorted()
        return String(Hashing.sha256Hex(lines.joined(separator: "\n")).prefix(16))
    }
}

/// Finds modules that read the conditioning signal (e.g. text embeddings) — without
/// naming any architecture. Cross-attention K/V projections have a telltale structure:
/// within one component, they read a width that no module in that component writes
/// ("read-only width"), and they come in sibling pairs with identical shapes. When that
/// structure is absent (e.g. joint-attention transformers), fall back to attention-like
/// projection names, then to all modules. Always reported as a heuristic.
public struct ConditioningHeuristic: Sendable {
    public enum Method: String, Codable, Sendable {
        case readOnlyWidth = "read-only-width"
        case attentionNames = "attention-names"
        case all
    }

    public let method: Method
    public let conditioningDims: [Int]
    public let candidates: Set<String>

    public init(modules: [ModuleSpec]) {
        var byComponent: [String: [ModuleSpec]] = [:]
        for m in modules { byComponent[Self.component(of: m.name), default: []].append(m) }

        var dims: Set<Int> = []
        var picked: Set<String> = []
        for (_, group) in byComponent {
            var inCount: [Int: Int] = [:], outCount: [Int: Int] = [:]
            for m in group {
                inCount[m.inDim, default: 0] += 1
                outCount[m.outDim, default: 0] += 1
            }
            let readOnly = Set(inCount.filter { $0.value >= 2 && outCount[$0.key] == nil }.keys)
            guard !readOnly.isEmpty else { continue }
            // Sibling pairs: same parent path, same (out, in), reading a read-only width.
            var byParent: [String: [ModuleSpec]] = [:]
            for m in group where readOnly.contains(m.inDim) {
                byParent[Self.parent(of: m.name), default: []].append(m)
            }
            for (_, sibs) in byParent where sibs.count >= 2 {
                let shapes = Dictionary(grouping: sibs) { "\($0.outDim)x\($0.inDim)" }
                for (_, same) in shapes where same.count >= 2 {
                    for m in same {
                        picked.insert(m.name)
                        dims.insert(m.inDim)
                    }
                }
            }
        }
        if !picked.isEmpty {
            method = .readOnlyWidth
            conditioningDims = dims.sorted()
            candidates = picked
            return
        }
        let attention = modules.filter { Self.isAttentionLike($0.name) }.map(\.name)
        if !attention.isEmpty {
            method = .attentionNames
            conditioningDims = []
            candidates = Set(attention)
            return
        }
        method = .all
        conditioningDims = []
        candidates = Set(modules.map(\.name))
    }

    /// Component = adapter-convention prefix ("lora_unet", "lora_te1") or first path segment.
    static func component(of name: String) -> String {
        if name.hasPrefix("lora_"), let r = name.range(of: "^lora_[a-z]+[0-9]*", options: .regularExpression) {
            return String(name[r])
        }
        if let dot = name.firstIndex(of: ".") { return String(name[..<dot]) }
        return ""
    }

    /// Parent path: drop the last token (dotted, or kohya underscore form).
    static func parent(of name: String) -> String {
        let tokens = tokenize(name)
        return tokens.dropLast().joined(separator: ".")
    }

    static func tokenize(_ name: String) -> [String] {
        name.split(whereSeparator: { $0 == "." || $0 == "_" }).map(String.init)
    }

    static let keyValueTokens: Set<String> = ["k", "v", "key", "value", "qkv", "kv", "wk", "wv"]

    static func isAttentionLike(_ name: String) -> Bool {
        let tokens = tokenize(name.lowercased())
        guard let last = tokens.last else { return false }
        if keyValueTokens.contains(last) { return true }
        // "to_k", "k_proj", "add_k_proj", "in_proj" (fused qkv)
        let tail = tokens.suffix(3)
        guard last == "proj" else { return false }
        return tail.contains { keyValueTokens.contains($0) } || tail.contains("in")
    }
}
