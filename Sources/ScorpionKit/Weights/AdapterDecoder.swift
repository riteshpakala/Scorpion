//
//  AdapterDecoder.swift
//  ScorpionKit
//
//  Recognizes adapter file conventions by tensor-name suffix — these conventions are
//  shared across architectures (the same kohya/PEFT/LyCORIS keys appear in SD, SDXL, Flux,
//  video and LLM adapters), so no architecture knowledge is needed — and reconstructs
//  each module's weight update ΔW:
//    LoRA  (kohya lora_up/lora_down/alpha, PEFT lora_B/lora_A)  ΔW = (α/r)·up·down
//    LoHa  (hada_w1_a/b, hada_w2_a/b)                           ΔW = (w1a·w1b) ⊙ (w2a·w2b)·(α/r)
//    LoKr  (lokr_w1[_a,_b], lokr_w2[_a,_b])                     ΔW = kron(w1, w2)·scale
//    diff  (LyCORIS full diff)                                   ΔW = diff
//  A unit with no adapter keys is a full model; its 2-D/conv weights are modules whose
//  ΔW is W − W_base when a base is available, else W itself.
//

import Foundation
import MLX

public enum AdapterFormat: String, Codable, Sendable {
    case lora
    case loha
    case lokr
    case diff
    case mixed
    case textualInversion
    case fullModel
}

public struct ModuleSpec: Sendable, Hashable {
    public enum Kind: Sendable, Hashable {
        case lora(up: String, down: String, alpha: String?)
        case loha(w1a: String, w1b: String, w2a: String, w2b: String, alpha: String?)
        case lokr(w1: String?, w1a: String?, w1b: String?, w2: String?, w2a: String?, w2b: String?, alpha: String?)
        case dense(weight: String)
    }

    public let name: String
    public let kind: Kind
    public let outDim: Int
    public let inDim: Int
    /// Adapter rank, when the format has one.
    public let rank: Int?
    /// DoRA magnitude present (direction-only approximation is used).
    public let hasDoRA: Bool
    /// Tucker/mid factors present but ignored.
    public let ignoredFactors: [String]

    public var tensorNames: [String] {
        var names: [String]
        switch kind {
        case .lora(let up, let down, let alpha): names = [up, down] + (alpha.map { [$0] } ?? [])
        case .loha(let a, let b, let c, let d, let alpha): names = [a, b, c, d] + (alpha.map { [$0] } ?? [])
        case .lokr(let w1, let w1a, let w1b, let w2, let w2a, let w2b, let alpha):
            names = [w1, w1a, w1b, w2, w2a, w2b, alpha].compactMap { $0 }
        case .dense(let w): names = [w]
        }
        return names
    }

    public var isAdapter: Bool {
        if case .dense = kind { return false }
        return true
    }
}

public struct AdapterLayout: Sendable {
    public let format: AdapterFormat
    public let modules: [ModuleSpec]
    /// Tensor names that matched no module (norm weights, biases, embeddings…).
    public let unassigned: [String]

    public var totalRankParameters: Int {
        modules.reduce(0) { acc, m in acc + (m.rank.map { $0 * (m.outDim + m.inDim) } ?? m.outDim * m.inDim) }
    }
}

public enum AdapterLayoutDetector {
    enum Role: String {
        case loraUp, loraDown, loraMid, alpha, dora
        case hadaW1A, hadaW1B, hadaW2A, hadaW2B, hadaT1, hadaT2
        case lokrW1, lokrW1A, lokrW1B, lokrW2, lokrW2A, lokrW2B, lokrT2
        case diff, diffBias
    }

    /// Longest suffixes first so ".lora_A.default.weight" wins over ".weight".
    static let suffixes: [(String, Role)] = [
        (".lora_A.default.weight", .loraDown), (".lora_B.default.weight", .loraUp),
        ("_lora.down.weight", .loraDown), ("_lora.up.weight", .loraUp),
        (".lora.down.weight", .loraDown), (".lora.up.weight", .loraUp),
        (".lora_down.weight", .loraDown), (".lora_up.weight", .loraUp), (".lora_mid.weight", .loraMid),
        (".lora_A.weight", .loraDown), (".lora_B.weight", .loraUp),
        (".lora_magnitude_vector", .dora), (".dora_scale", .dora),
        (".hada_w1_a", .hadaW1A), (".hada_w1_b", .hadaW1B), (".hada_w2_a", .hadaW2A), (".hada_w2_b", .hadaW2B),
        (".hada_t1", .hadaT1), (".hada_t2", .hadaT2),
        (".lokr_w1_a", .lokrW1A), (".lokr_w1_b", .lokrW1B), (".lokr_w2_a", .lokrW2A), (".lokr_w2_b", .lokrW2B),
        (".lokr_w1", .lokrW1), (".lokr_w2", .lokrW2), (".lokr_t2", .lokrT2),
        (".diff_b", .diffBias), (".diff", .diff),
        (".alpha", .alpha),
    ]

    static func split(_ name: String) -> (module: String, role: Role)? {
        for (suffix, role) in suffixes where name.hasSuffix(suffix) {
            return (String(name.dropLast(suffix.count)), role)
        }
        return nil
    }

    public static func detect(_ tensors: [String: TensorRecord]) -> AdapterLayout {
        var groups: [String: [Role: String]] = [:]
        var unassigned: [String] = []
        for name in tensors.keys {
            if let (module, role) = split(name) {
                groups[module, default: [:]][role] = name
            } else {
                unassigned.append(name)
            }
        }
        let hasAdapterKeys = groups.values.contains { roles in
            roles.keys.contains { $0 != .alpha && $0 != .dora }
        }

        if !hasAdapterKeys {
            if let ti = textualInversion(tensors) { return ti }
            return fullModel(tensors)
        }

        var modules: [ModuleSpec] = []
        var formats: Set<AdapterFormat> = []
        func shape(_ n: String?) -> [Int]? { n.flatMap { tensors[$0]?.shape } }
        func flat(_ s: [Int]) -> Int { s.dropFirst().reduce(1, *) }

        for (module, roles) in groups {
            let dora = roles[.dora] != nil
            if let up = roles[.loraUp], let down = roles[.loraDown],
               let us = shape(up), let ds = shape(down), us.count >= 2, ds.count >= 2 {
                formats.insert(.lora)
                modules.append(ModuleSpec(name: module, kind: .lora(up: up, down: down, alpha: roles[.alpha]),
                                          outDim: us[0], inDim: flat(ds), rank: ds[0], hasDoRA: dora,
                                          ignoredFactors: roles[.loraMid] != nil ? ["lora_mid"] : []))
            } else if let a = roles[.hadaW1A], let b = roles[.hadaW1B], let c = roles[.hadaW2A], let d = roles[.hadaW2B],
                      let sa = shape(a), let sb = shape(b), sa.count >= 2, sb.count >= 2 {
                formats.insert(.loha)
                modules.append(ModuleSpec(name: module, kind: .loha(w1a: a, w1b: b, w2a: c, w2b: d, alpha: roles[.alpha]),
                                          outDim: sa[0], inDim: flat(sb), rank: sb[0], hasDoRA: dora,
                                          ignoredFactors: roles[.hadaT1] != nil ? ["hada_t1", "hada_t2"] : []))
            } else if roles[.lokrW1] != nil || roles[.lokrW1A] != nil {
                let w1 = shape(roles[.lokrW1]) ?? [shape(roles[.lokrW1A])?.first ?? 0, shape(roles[.lokrW1B])?.last ?? 0]
                let w2: [Int]
                if let s = shape(roles[.lokrW2]) {
                    w2 = [s[0], flat(s)]
                } else {
                    w2 = [shape(roles[.lokrW2A])?.first ?? 0, flat(shape(roles[.lokrW2B]) ?? [0, 0])]
                }
                guard w1.count >= 2, w1[0] > 0, w2[0] > 0 else { continue }
                formats.insert(.lokr)
                let rank = shape(roles[.lokrW2B])?.first ?? shape(roles[.lokrW1B])?.first
                modules.append(ModuleSpec(name: module,
                                          kind: .lokr(w1: roles[.lokrW1], w1a: roles[.lokrW1A], w1b: roles[.lokrW1B],
                                                      w2: roles[.lokrW2], w2a: roles[.lokrW2A], w2b: roles[.lokrW2B],
                                                      alpha: roles[.alpha]),
                                          outDim: w1[0] * w2[0], inDim: w1[1] * w2[1], rank: rank, hasDoRA: dora,
                                          ignoredFactors: roles[.lokrT2] != nil ? ["lokr_t2"] : []))
            } else if let diff = roles[.diff], let s = shape(diff), s.count >= 2 {
                formats.insert(.diff)
                modules.append(ModuleSpec(name: module, kind: .dense(weight: diff), outDim: s[0], inDim: flat(s),
                                          rank: nil, hasDoRA: false, ignoredFactors: []))
            } else {
                unassigned.append(contentsOf: roles.values)
            }
        }
        let format: AdapterFormat = formats.count == 1 ? formats.first! : (formats.isEmpty ? .fullModel : .mixed)
        return AdapterLayout(format: format, modules: modules.sorted { $0.name.naturalLess($1.name) },
                             unassigned: unassigned.sorted())
    }

    /// Textual-inversion embeddings: a handful of small 2-D tensors, no module structure.
    static func textualInversion(_ tensors: [String: TensorRecord]) -> AdapterLayout? {
        guard tensors.count <= 4, !tensors.isEmpty else { return nil }
        let known = ["emb_params", "string_to_param", "clip_l", "clip_g", "t5"]
        let matches = tensors.values.allSatisfy { t in
            t.shape.count == 2 && t.shape[0] <= 128 && known.contains { t.name.lowercased().contains($0) }
        }
        guard matches else { return nil }
        let modules = tensors.values.map {
            ModuleSpec(name: $0.name, kind: .dense(weight: $0.name), outDim: $0.shape[0], inDim: $0.shape[1],
                       rank: nil, hasDoRA: false, ignoredFactors: [])
        }
        return AdapterLayout(format: .textualInversion, modules: modules.sorted { $0.name < $1.name }, unassigned: [])
    }

    static func fullModel(_ tensors: [String: TensorRecord]) -> AdapterLayout {
        var modules: [ModuleSpec] = []
        var unassigned: [String] = []
        for t in tensors.values {
            if t.name.hasSuffix("weight"), t.shape.count >= 2, t.shape[0] > 1, t.dtype?.isFloatingPoint ?? false {
                modules.append(ModuleSpec(name: String(t.name.dropLast(t.name.hasSuffix(".weight") ? 7 : 0)),
                                          kind: .dense(weight: t.name), outDim: t.shape[0],
                                          inDim: t.shape.dropFirst().reduce(1, *), rank: nil,
                                          hasDoRA: false, ignoredFactors: []))
            } else {
                unassigned.append(t.name)
            }
        }
        return AdapterLayout(format: .fullModel, modules: modules.sorted { $0.name.naturalLess($1.name) },
                             unassigned: unassigned.sorted())
    }
}

// MARK: - Decoding

/// A module's weight update, kept factored when possible.
public enum WeightUpdate {
    case lowRank(up: MLXArray, down: MLXArray, scale: Float)   // up (out×r), down (r×in)
    case dense(MLXArray)                                       // (out×in)

    public var materialized: MLXArray {
        switch self {
        case .lowRank(let up, let down, let scale): return matmul(up, down) * scale
        case .dense(let w): return w
        }
    }
}

public enum AdapterDecoderError: Error, LocalizedError {
    case missingTensor(String)
    case shapeMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .missingTensor(let n): return "Missing tensor \(n)"
        case .shapeMismatch(let n): return "Shape mismatch in module \(n)"
        }
    }
}

public struct AdapterDecoder {
    /// PEFT `lora_alpha / r` (or `/√r` with rsLoRA) from adapter_config.json, when present.
    public var peftScale: ((Int) -> Float)?

    public init(peftScale: ((Int) -> Float)? = nil) { self.peftScale = peftScale }

    /// Parse PEFT adapter_config.json into a rank → scale function.
    public static func peftScale(fromConfig data: Data) -> ((Int) -> Float)? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let alpha = (json["lora_alpha"] as? NSNumber)?.floatValue else { return nil }
        let rs = json["use_rslora"] as? Bool ?? false
        let fixedR = (json["r"] as? NSNumber)?.intValue
        return { r in
            let rank = Float(fixedR ?? r)
            return rs ? alpha / rank.squareRoot() : alpha / rank
        }
    }

    /// Reconstruct ΔW for `spec` from decoded tensors (float32, original shapes).
    public func decode(_ spec: ModuleSpec, tensors: [String: MLXArray]) throws -> WeightUpdate {
        func t(_ name: String?) throws -> MLXArray {
            guard let name else { throw AdapterDecoderError.missingTensor("(nil) in \(spec.name)") }
            guard let a = tensors[name] else { throw AdapterDecoderError.missingTensor(name) }
            return a
        }
        func as2D(_ a: MLXArray) -> MLXArray { a.reshaped(a.dim(0), -1) }
        func alphaScale(_ alphaName: String?, rank: Int) -> Float? {
            guard let alphaName, let a = tensors[alphaName] else { return nil }
            return a.reshaped(-1)[0].item(Float.self) / Float(max(rank, 1))
        }

        switch spec.kind {
        case .lora(let upName, let downName, let alphaName):
            let up = as2D(try t(upName))
            let down = as2D(try t(downName))
            let r = down.dim(0)
            guard up.dim(1) == r else { throw AdapterDecoderError.shapeMismatch(spec.name) }
            let scale = alphaScale(alphaName, rank: r) ?? peftScale?(r) ?? 1
            return .lowRank(up: up, down: down, scale: scale)

        case .loha(let a, let b, let c, let d, let alphaName):
            let w1 = matmul(as2D(try t(a)), as2D(try t(b)))
            let w2 = matmul(as2D(try t(c)), as2D(try t(d)))
            guard w1.shape == w2.shape else { throw AdapterDecoderError.shapeMismatch(spec.name) }
            let r = try t(b).dim(0)
            return .dense(w1 * w2 * (alphaScale(alphaName, rank: r) ?? 1))

        case .lokr(let w1n, let w1a, let w1b, let w2n, let w2a, let w2b, let alphaName):
            let w1 = try w1n.map { as2D(try t($0)) } ?? matmul(as2D(try t(w1a)), as2D(try t(w1b)))
            let w2 = try w2n.map { as2D(try t($0)) } ?? matmul(as2D(try t(w2a)), as2D(try t(w2b)))
            let factorized = w1n == nil || w2n == nil
            let r = spec.rank ?? 1
            let scale = factorized ? (alphaScale(alphaName, rank: r) ?? 1) : 1
            return .dense(Self.kron(w1, w2) * scale)

        case .dense(let name):
            return .dense(as2D(try t(name)))
        }
    }

    /// Kronecker product of 2-D arrays: (a×c) ⊗ (b×d) → (ab × cd).
    public static func kron(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let (ar, ac) = (a.dim(0), a.dim(1))
        let (br, bc) = (b.dim(0), b.dim(1))
        let out = a.reshaped(ar, 1, ac, 1) * b.reshaped(1, br, 1, bc)
        return out.reshaped(ar * br, ac * bc)
    }
}
