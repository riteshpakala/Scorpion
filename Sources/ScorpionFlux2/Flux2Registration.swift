//
//  Flux2Registration.swift
//  ScorpionFlux2
//
//  `--backend flux2-klein`: builds the model pair for a Klein LoRA. The adapter is extracted
//  by ScorpionKit's format-level reader (a Hugging Face / GitHub link or a local
//  .safetensors), mapped onto FluxKit's linears, and applied at run time through the
//  transformer's linear hook — one transformer, adapter on for the target, off for the base.
//
//  Options (`--option key=value`):
//    base-dir     mflux export with text_encoder/, vae/, tokenizer/ (default: Flux2Weights.defaultDirectory)
//    transformer  "base" (download FLUX.2-klein-base-4B, the default) | "mflux" (base-dir/transformer,
//                 the distilled 4-bit export — plumbing only) | a path to a diffusers transformer
//    quantize     4 | 8 — re-quantize a diffusers transformer on load (default: keep bf16)
//    long-side    reference resize, multiple of 16 (default 512)
//    max-batch    rows per forward (default 4)
//

import FluxKit
import Foundation
import MLX
import MLXNN
import ScorpionKit

public enum Flux2Registration {
    public static let prefix = "flux2-klein"

    public static func register(in registry: BackendRegistry = .shared) {
        registry.register(prefix: prefix) { try await Flux2KleinFactory.make($0) }
    }
}

public enum Flux2KleinFactory {
    public static func make(_ request: BackendRequest) async throws -> ModelPair {
        let dir = request.option("base-dir").map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
            ?? Flux2Weights.defaultDirectory
        let choice = request.option("transformer") ?? "base"
        try Flux2Weights.validate(dir, needsTransformer: choice == "mflux")
        var provenance: [String: String] = ["family": "FLUX.2 Klein 4B", "components": dir.path]
        var issues: [String] = []

        // 1. The base transformer's tensors (FluxKit keys).
        let store: TensorStore
        let variant: String
        switch choice {
        case "mflux":
            store = try TensorStore(componentDir: dir.appendingPathComponent("transformer"))
            variant = Flux2Weights.variant(of: dir) ?? "unknown"
            provenance["transformer"] = dir.appendingPathComponent("transformer").path + " (mflux, 4-bit)"
        default:
            let url: URL
            if choice == "base" {
                url = try await Flux2Weights.baseTransformer()
                variant = "base"
            } else {
                url = URL(fileURLWithPath: (choice as NSString).expandingTildeInPath)
                variant = url.path.lowercased().contains("klein-base") ? "base" : (Flux2Weights.variant(of: url) ?? "unknown")
            }
            let bits = request.option("quantize").flatMap(Int.init)
            store = TensorStore(tensors: try Flux2Weights.diffusersTransformer(at: url, quantizeBits: bits))
            provenance["transformer"] = url.path + (bits.map { " (diffusers, re-quantized to \($0)-bit)" } ?? " (diffusers, bf16)")
        }
        provenance["transformer-variant"] = variant
        if variant != "base" {
            issues.append("the \(variant) Klein transformer only evaluates its few sampling noise levels; the probe needs σ ≤ 0.5 "
                          + "(use --option transformer=base, FLUX.2-klein-base-4B)")
        }

        // 2. The adapter under test, mapped onto FluxKit's linears.
        let adapterSwitch = AdapterSwitch()
        var mapped: [String: WeightUpdate] = [:]
        var triggers: [String] = []
        if let link = request.model {
            let adapter = try await Scorpion(options: ScorpionOptions()).extractAdapter(model: link)
            triggers = adapter.findings.triggerWords
            provenance["adapter"] = adapter.source.link
            provenance["adapter-pin"] = adapter.source.commit ?? "unpinned"
            provenance["adapter-format"] = "\(adapter.format.rawValue), \(adapter.updates.count) modules"
            if let base = adapter.findings.baseModel {
                provenance["adapter-base-model"] = base
                let b = base.lowercased()
                if !b.contains("klein") {
                    issues.append("the adapter was trained on \(base), not FLUX.2 Klein")
                } else if b.contains("base") != (variant == "base") {
                    issues.append("the adapter was trained on \(base) but the probe runs the \(variant) transformer")
                }
            } else {
                provenance["adapter-base-model"] = "not stated by the adapter"
            }
            for n in adapter.notes { provenance["adapter-note-\(provenance.count)"] = n }
            let keys = Set(store.keys.filter { $0.hasSuffix(".weight") }.map { String($0.dropLast(".weight".count)) })
            let mapping = Flux2LoRAMapper.map(adapter.updates, linearKeys: keys)
            mapped = mapping.updates
            if !mapping.unmapped.isEmpty {
                issues.append("\(mapping.unmapped.count) adapter module(s) could not be placed on the transformer "
                              + "(e.g. \(mapping.unmapped.prefix(3).joined(separator: ", ")))")
            }
            if !mapping.unsupported.isEmpty {
                issues.append("\(mapping.unsupported.count) text-encoder module(s) are not applied by this executor")
            }
            if mapping.translated > 0 { provenance["adapter-mapping"] = "\(mapping.translated) BFL-native module(s) translated" }
        } else {
            provenance["adapter"] = "none — target = base (plumbing check: every map must read exactly zero)"
        }

        // 3. Build once, wrapping mapped linears; every mapped module must land, with matching shape.
        var applied = Set<String>(), mismatched: [String] = []
        store.linearTransform = { prefix, layer in
            guard let update = mapped[prefix] else { return layer }
            let delta = DeltaLinear(base: layer, update: update, switch: adapterSwitch)
            guard delta.deltaShape == layer.logicalShape else {
                mismatched.append("\(prefix) \(delta.deltaShape) vs \(layer.logicalShape)")
                return layer
            }
            applied.insert(prefix)
            return delta
        }
        let transformer = try Flux2Transformer(store: store)
        let missed = Set(mapped.keys).subtracting(applied).subtracting(mismatched.map { String($0.split(separator: " ")[0]) })
        if !mismatched.isEmpty { issues.append("\(mismatched.count) adapter module(s) have the wrong shape (e.g. \(mismatched[0]))") }
        if !missed.isEmpty { issues.append("\(missed.count) mapped module(s) are not linears the transformer builds") }
        if request.model != nil { provenance["adapter-applied"] = "\(applied.count) linear(s)" }

        let longSide = Int(request.option("long-side") ?? "") ?? 512
        let maxBatch = max(1, Int(request.option("max-batch") ?? "") ?? 4)
        let model = Flux2Model(directory: dir, transformer: transformer, adapterSwitch: adapterSwitch, longSide: longSide,
                               transformerSource: provenance["transformer"] ?? "")
        // Pre-encode the prompts the probe will use, in one text-encoder session.
        let prompt = ProbePrompt.resolve(user: request.prompt, triggers: triggers)
        _ = try model.embeddings(for: Array(Set([prompt.text, ProbePrompt.unconditional])))
        provenance["resolution"] = "long side \(model.longSide) px"
        let targetID = request.model.map { "klein-\(variant)+\(URL(fileURLWithPath: $0).lastPathComponent)" } ?? "klein-\(variant)"
        return ModelPair(target: Flux2KleinBackend(identifier: targetID, model: model, adapterOn: request.model != nil, maxBatch: maxBatch),
                         base: Flux2KleinBackend(identifier: "klein-\(variant)", model: model, adapterOn: false, maxBatch: maxBatch),
                         provenance: provenance, issues: issues, triggers: triggers)
    }
}
