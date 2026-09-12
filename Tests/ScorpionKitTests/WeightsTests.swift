import Foundation
import MLX
import Testing
@testable import ScorpionKit

/// Synthetic adapter files shaped like cross-attention LoRAs, without naming any model.
enum AdapterFixtures {
    static func record(_ name: String, _ shape: [Int]) -> TensorRecord {
        TensorRecord(name: name, file: RemoteFile(path: "x.safetensors", size: nil, url: URL(string: "https://x.test")!),
                     rawDType: "F32", shape: shape, byteRange: 0..<(4 * shape.reduce(1, *)))
    }

    /// Blocks with a residual width that is both read and written, plus K/V projections that
    /// read a width (`cond`) nothing in the component writes.
    static func crossAttentionLayout(blocks: Int = 3, width: Int = 32, cond: Int = 24, rank: Int = 4,
                                     prefix: String = "lora_unet") -> [String: TensorRecord] {
        var t: [String: TensorRecord] = [:]
        for b in 0..<blocks {
            for (proj, inDim) in [("to_q", width), ("to_k", cond), ("to_v", cond), ("to_out_0", width)] {
                let m = "\(prefix)_blocks_\(b)_attn2_\(proj)"
                t["\(m).lora_down.weight"] = record("\(m).lora_down.weight", [rank, inDim])
                t["\(m).lora_up.weight"] = record("\(m).lora_up.weight", [width, rank])
                t["\(m).alpha"] = record("\(m).alpha", [])
            }
            let ff = "\(prefix)_blocks_\(b)_ff_net_0_proj"
            t["\(ff).lora_down.weight"] = record("\(ff).lora_down.weight", [rank, width])
            t["\(ff).lora_up.weight"] = record("\(ff).lora_up.weight", [4 * width, rank])
        }
        return t
    }
}

struct AdapterLayoutTests {
    @Test func detectsKohyaLoRAAndDims() {
        let layout = AdapterLayoutDetector.detect(AdapterFixtures.crossAttentionLayout())
        #expect(layout.format == .lora)
        #expect(layout.modules.count == 15)
        let k = layout.modules.first { $0.name.hasSuffix("blocks_0_attn2_to_k") }!
        #expect(k.outDim == 32 && k.inDim == 24 && k.rank == 4)
        #expect(k.tensorNames.count == 3)
    }

    @Test func detectsPEFTLoHaLoKrAndFullModels() {
        let r = AdapterFixtures.record
        let peft = AdapterLayoutDetector.detect([
            "unet.a.to_k.lora_A.weight": r("unet.a.to_k.lora_A.weight", [4, 16]),
            "unet.a.to_k.lora_B.weight": r("unet.a.to_k.lora_B.weight", [8, 4]),
        ])
        #expect(peft.format == .lora && peft.modules.first?.inDim == 16)

        let loha = AdapterLayoutDetector.detect([
            "m.hada_w1_a": r("m.hada_w1_a", [8, 2]), "m.hada_w1_b": r("m.hada_w1_b", [2, 6]),
            "m.hada_w2_a": r("m.hada_w2_a", [8, 2]), "m.hada_w2_b": r("m.hada_w2_b", [2, 6]),
        ])
        #expect(loha.format == .loha && loha.modules.first?.outDim == 8 && loha.modules.first?.inDim == 6)

        let lokr = AdapterLayoutDetector.detect([
            "m.lokr_w1": r("m.lokr_w1", [2, 3]), "m.lokr_w2_a": r("m.lokr_w2_a", [4, 2]), "m.lokr_w2_b": r("m.lokr_w2_b", [2, 5]),
        ])
        #expect(lokr.format == .lokr && lokr.modules.first?.outDim == 8 && lokr.modules.first?.inDim == 15)

        let full = AdapterLayoutDetector.detect([
            "blk.0.attn.weight": r("blk.0.attn.weight", [8, 8]), "blk.0.norm.weight": r("blk.0.norm.weight", [8]),
        ])
        #expect(full.format == .fullModel && full.modules.count == 1 && full.unassigned == ["blk.0.norm.weight"])
    }
}

struct AdapterDecoderTests {
    func array(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXArray(Fixtures.randomFloats(shape.reduce(1, *), seed: seed), shape)
    }

    @Test func loraScaleAndDenseAgree() throws {
        let up = array([8, 3], seed: 1), down = array([3, 6], seed: 2)
        let spec = ModuleSpec(name: "m", kind: .lora(up: "u", down: "d", alpha: "a"), outDim: 8, inDim: 6, rank: 3,
                              hasDoRA: false, ignoredFactors: [])
        let update = try AdapterDecoder().decode(spec, tensors: ["u": up, "d": down, "a": MLXArray([Float(1.5)])])
        #expect(maxAbsDiff(update.materialized, matmul(up, down) * 0.5) < 1e-5)
    }

    @Test func peftScaleFromConfig() throws {
        let scale = AdapterDecoder.peftScale(fromConfig: Data(#"{"lora_alpha": 16, "r": 8}"#.utf8))!
        #expect(scale(8) == 2)
        let rs = AdapterDecoder.peftScale(fromConfig: Data(#"{"lora_alpha": 16, "r": 4, "use_rslora": true}"#.utf8))!
        #expect(rs(4) == 8)
    }

    @Test func lohaAndLokrReconstruction() throws {
        let a = array([4, 2], seed: 3), b = array([2, 5], seed: 4), c = array([4, 2], seed: 5), d = array([2, 5], seed: 6)
        let loha = ModuleSpec(name: "h", kind: .loha(w1a: "a", w1b: "b", w2a: "c", w2b: "d", alpha: nil), outDim: 4, inDim: 5,
                              rank: 2, hasDoRA: false, ignoredFactors: [])
        let got = try AdapterDecoder().decode(loha, tensors: ["a": a, "b": b, "c": c, "d": d]).materialized
        #expect(maxAbsDiff(got, matmul(a, b) * matmul(c, d)) < 1e-5)

        let w1 = MLXArray([Float(1), 2, 3, 4], [2, 2]), w2 = MLXArray([Float(0), 1, 1, 0], [2, 2])
        let k = AdapterDecoder.kron(w1, w2).asArray(Float.self)
        #expect(k == [0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0])
    }
}

struct SpectralTests {
    @Test func lowRankPathMatchesDenseSpectrum() throws {
        let up = MLXArray(Fixtures.randomFloats(40 * 5, seed: 11), [40, 5])
        let down = MLXArray(Fixtures.randomFloats(5 * 30, seed: 12), [5, 30])
        let lr = SpectralAnalyzer.analyze(module: "m", update: .lowRank(up: up, down: down, scale: 0.7))!
        let dense = SpectralAnalyzer.analyze(module: "m", update: .dense(matmul(up, down) * 0.7))!
        for (a, b) in zip(lr.summary.topSingularValues, dense.summary.topSingularValues.prefix(5)) {
            #expect(abs(a - b) / max(b, 1e-6) < 1e-3)
        }
        #expect(abs(lr.summary.frobeniusNorm - dense.summary.frobeniusNorm) / dense.summary.frobeniusNorm < 1e-3)
        // u₁ agrees up to sign, and is sign-canonical in both.
        let cos = abs((lr.u1 * dense.u1).sum().item(Float.self))
        #expect(cos > 0.999)
    }

    @Test func entropyExtremes() {
        let u = MLXArray(Fixtures.randomFloats(16, seed: 1), [16, 1]), v = MLXArray(Fixtures.randomFloats(12, seed: 2), [1, 12])
        let rank1 = SpectralAnalyzer.analyze(module: "r1", update: .dense(matmul(u, v)))!.summary
        #expect(rank1.concentration > 0.999 && rank1.effectiveRank < 1.01 && rank1.normalizedEntropy < 0.01)
        let iso = SpectralAnalyzer.analyze(module: "iso", update: .dense(MLXArray.eye(8)))!.summary
        #expect(abs(iso.effectiveRank - 8) < 1e-3 && iso.normalizedEntropy > 0.999)
    }
}

struct ConditioningHeuristicTests {
    @Test func findsReadOnlyWidthPairs() {
        let layout = AdapterLayoutDetector.detect(AdapterFixtures.crossAttentionLayout())
        let h = ConditioningHeuristic(modules: layout.modules)
        #expect(h.method == .readOnlyWidth)
        #expect(h.conditioningDims == [24])
        #expect(h.candidates.count == 6)
        #expect(h.candidates.allSatisfy { $0.hasSuffix("to_k") || $0.hasSuffix("to_v") })
    }

    @Test func componentsAreSeparated() {
        // A text-encoder component writing width 24 must not hide the denoiser's read-only 24.
        var tensors = AdapterFixtures.crossAttentionLayout()
        let te = "lora_te_text_model_encoder_layers_0_mlp_fc1"
        tensors["\(te).lora_down.weight"] = AdapterFixtures.record("\(te).lora_down.weight", [4, 24])
        tensors["\(te).lora_up.weight"] = AdapterFixtures.record("\(te).lora_up.weight", [24, 4])
        let h = ConditioningHeuristic(modules: AdapterLayoutDetector.detect(tensors).modules)
        #expect(h.method == .readOnlyWidth && h.candidates.count == 6)
    }

    @Test func fallsBackToAttentionNames() {
        let r = AdapterFixtures.record
        let modules = AdapterLayoutDetector.detect([
            "t.b0.attn.to_k.lora_A.weight": r("t.b0.attn.to_k.lora_A.weight", [2, 16]),
            "t.b0.attn.to_k.lora_B.weight": r("t.b0.attn.to_k.lora_B.weight", [16, 2]),
            "t.b0.ff.lora_A.weight": r("t.b0.ff.lora_A.weight", [2, 16]),
            "t.b0.ff.lora_B.weight": r("t.b0.ff.lora_B.weight", [16, 2]),
        ]).modules
        let h = ConditioningHeuristic(modules: modules)
        #expect(h.method == .attentionNames && h.candidates == ["t.b0.attn.to_k"])
    }

    @Test func familyKeyIgnoresRankAndBlockIndex() {
        let a = AdapterLayoutDetector.detect(AdapterFixtures.crossAttentionLayout(rank: 4)).modules
        let b = AdapterLayoutDetector.detect(AdapterFixtures.crossAttentionLayout(rank: 16)).modules
        let c = AdapterLayoutDetector.detect(AdapterFixtures.crossAttentionLayout(cond: 48)).modules
        #expect(StructuralHash.familyKey(a) == StructuralHash.familyKey(b))
        #expect(StructuralHash.familyKey(a) != StructuralHash.familyKey(c))
        #expect(StructuralHash.pattern("blocks_12_attn2") == "blocks_#_attn#")
    }
}

struct MetadataProbeTests {
    @Test func kohyaTagsTriggersAndIndicators() {
        let meta = [
            "ss_tag_frequency": #"{"10_ohwx": {"ohwx woman": 20, "red_hair": 15, "smile": 5}, "5_reg": {"woman": 3}}"#,
            "ss_dataset_dirs": #"{"10_ohwx": {"n_repeats": 10, "img_count": 20}}"#,
            "modelspec.trigger_phrase": "ohwx woman",
            "ss_base_model_version": "sdxl_base_v1-0",
        ]
        let f = MetadataProbe.findings(safetensorsMetadata: meta, cardData: [:], readme: nil, hubTags: ["lora"])
        #expect(f.trainingTags.first?.tag == "ohwx woman")
        #expect(f.trainingTags.contains { $0.tag == "red hair" && $0.count == 15 })
        #expect(f.trainingTagTotal == 43)
        #expect(f.trainingImageCount == 20)
        #expect(f.triggerWords == ["ohwx woman"])
        #expect(f.baseModel == "sdxl_base_v1-0")
        #expect(f.personIndicators.contains("woman"))
    }

    @Test func frontMatterAndReadmeTriggers() {
        let readme = """
        ---
        base_model: stabilityai/stable-diffusion-xl-base-1.0
        tags:
          - lora
          - portrait
        instance_prompt: a photo of sks person
        ---
        # My Portrait LoRA
        Trigger word: `sks person`
        """
        let card = MetadataProbe.frontMatter(readme)
        #expect(card["base_model"]?.stringValue == "stabilityai/stable-diffusion-xl-base-1.0")
        #expect(card["tags"]?.flattenedStrings == ["lora", "portrait"])
        let f = MetadataProbe.findings(safetensorsMetadata: [:], cardData: card, readme: readme, hubTags: [])
        #expect(f.triggerWords.contains("a photo of sks person"))
        #expect(f.triggerWords.contains("sks person"))
        #expect(f.title == "My Portrait LoRA")
        #expect(f.personIndicators.contains("portrait"))
    }

    @Test func normalization() {
        #expect(MetadataProbe.normalize("(Red_Hair:1.2)") == "red hair")
        #expect(MetadataProbe.normalize("  Long   hair, ") == "long hair")
    }
}

@Suite(.serialized) struct WeightAnalyzerTests {
    /// A remote LoRA served by the stub: only conditioning readers should be fetched.
    @Test func analyzesRemoteAdapterFetchingOnlyConditioningReaders() async throws {
        var tensors: [SafetensorsWriter.Tensor] = []
        var seed: UInt64 = 100
        for (name, rec) in AdapterFixtures.crossAttentionLayout().sorted(by: { $0.key < $1.key }) {
            seed += 1
            let shape = rec.shape.isEmpty ? [1] : rec.shape
            let values = rec.shape.isEmpty ? [Float(4)] : Fixtures.randomFloats(shape.reduce(1, *), seed: seed)
            tensors.append(.init(name: name, floats: values, shape: rec.shape, dtype: .f16))
        }
        let data = try SafetensorsWriter.encode(tensors, metadata: ["ss_output_name": "fixture"])
        let url = URL(string: "https://analyzer.test/lora.safetensors")!
        StubProtocol.register(url, .init(data: data))

        let fetcher = RangeFetcher.stubbed()
        let file = RemoteFile(path: "lora.safetensors", size: data.count, url: url)
        let header = try await InventoryReader(fetcher: fetcher).readSafetensorsHeader(file, cacheable: false)
        let unit = InventoryReader(fetcher: fetcher).makeUnit(name: file.path, files: [file], headers: [file.path: header])
        let headerBytes = fetcher.ledger.networkBytes

        var analyzer = WeightAnalyzer(fetcher: fetcher, budgetBytes: 1 << 30)
        analyzer.gapTolerance = 0
        let analysis = try await analyzer.analyze(unit, cacheable: false)
        let s = analysis.summary
        #expect(s.format == .lora)
        #expect(s.conditioning.method == .readOnlyWidth)
        #expect(s.analyzedModules == 6)
        #expect(analysis.fingerprints.count == 6)
        #expect(s.aggregate != nil && s.aggregate!.narrowness >= 0 && s.aggregate!.narrowness <= 1)
        // Exactly the planned tensors crossed the network — the K/V readers, not the file.
        let tensorBytes = fetcher.ledger.networkBytes - headerBytes
        #expect(tensorBytes == s.bytesPlanned)
        #expect(tensorBytes < data.count / 2, "fetched \(tensorBytes) of \(data.count)")
    }
}
