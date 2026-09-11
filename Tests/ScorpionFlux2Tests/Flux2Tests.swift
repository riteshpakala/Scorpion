//
//  The Klein executor. Pure-math checks always run; checks on real weights run when the mflux
//  export is on disk (Flux2Weights.defaultDirectory or SCORPION_FLUX2_DIR) and skip otherwise.
//

import CoreGraphics
import FluxKit
import Foundation
import MLX
import MLXNN
import Testing
@testable import ScorpionFlux2
@testable import ScorpionKit

func randn(_ shape: [Int], _ seed: UInt64) -> MLXArray { MLXRandom.normal(shape, key: MLXRandom.key(seed)) }

struct Flux2MathTests {
    @Test func packingRoundTrips() {
        let x = randn([2, 32, 8, 6], 1)
        let p = Flux2Packing.pack(x)
        #expect(p.shape == [2, 12, 128])
        #expect(abs(Flux2Packing.unpack(p, packedH: 4, packedW: 3) - x).max().item(Float.self) == 0)
        // Channel c·4 + dy·2 + dx of token (hp, wp) is pixel (2hp + dy, 2wp + dx) of channel c.
        #expect(p[1, 2 * 3 + 1, 5 * 4 + 1 * 2 + 0].item(Float.self) == x[1, 5, 2 * 2 + 1, 2 * 1 + 0].item(Float.self))
    }

    @Test func noiseScheduleIsFlowMatching() {
        let lambdas = MLXArray([Float(-4), 0, 2, 6])
        let (a, s) = FlowSchedule.alphaSigma(lambdas)
        #expect(abs(a + s - 1).max().item(Float.self) < 1e-6)
        #expect(abs(log(a.square() / s.square()) - lambdas).max().item(Float.self) < 1e-4)
        #expect(abs(s[1].item(Float.self) - 0.5) < 1e-6)
    }

    @Test func deltaIsSkippedWhenOffAndExactWhenOn() {
        let base = Linear(weight: randn([6, 5], 2), bias: randn([6], 3))
        let up = randn([6, 2], 4), down = randn([2, 5], 5)
        let s = AdapterSwitch()
        let d = DeltaLinear(base: base, update: .lowRank(up: up, down: down, scale: 0.5), switch: s)
        let x = randn([3, 5], 6)
        #expect(abs(d(x) - base(x)).max().item(Float.self) == 0, "off: bitwise the base")
        s.enabled = true
        let expected = base(x) + matmul(x, (matmul(up, down) * 0.5).transposed())
        #expect(abs(d(x) - expected).max().item(Float.self) < 1e-4)
        #expect(d.deltaShape == base.logicalShape)
    }

    @Test func loraNamesMapOntoFluxKitLinears() {
        let keys: Set<String> = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v",
                                 "transformer_blocks.0.attn.to_out", "transformer_blocks.0.attn.add_q_proj",
                                 "transformer_blocks.0.attn.add_k_proj", "transformer_blocks.0.attn.add_v_proj",
                                 "single_transformer_blocks.3.attn.to_qkv_mlp_proj", "time_guidance_embed.linear_1",
                                 "norm_out.linear"]
        func lr(_ out: Int) -> WeightUpdate { .lowRank(up: randn([out, 2], UInt64(out)), down: randn([2, 4], 9), scale: 1) }
        let updates: [String: WeightUpdate] = [
            "transformer.transformer_blocks.0.attn.to_q": lr(4),                       // PEFT / diffusers
            "base_model.model.transformer_blocks.0.attn.to_out.0": lr(4),              // diffusers to_out.0
            "lora_unet_single_transformer_blocks_3_attn_to_qkv_mlp_proj": lr(4),       // kohya underscores
            "time_guidance_embed.timestep_embedder.linear_1": lr(4),
            "diffusion_model.double_blocks.0.txt_attn.qkv": lr(12),                    // BFL fused qkv
            "final_layer.adaLN_modulation.1": lr(8),                                   // BFL (shift, scale)
            "lora_te1_text_model_encoder_layers_0_q_proj": lr(4),                      // text encoder
            "transformer.nonexistent_block.linear": lr(4),
        ]
        let m = Flux2LoRAMapper.map(updates, linearKeys: keys)
        #expect(Set(m.updates.keys) == ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_out",
                                        "single_transformer_blocks.3.attn.to_qkv_mlp_proj", "time_guidance_embed.linear_1",
                                        "transformer_blocks.0.attn.add_q_proj", "transformer_blocks.0.attn.add_k_proj",
                                        "transformer_blocks.0.attn.add_v_proj", "norm_out.linear"])
        #expect(m.unsupported == ["lora_te1_text_model_encoder_layers_0_q_proj"])
        #expect(m.unmapped == ["transformer.nonexistent_block.linear"])
        // Fused qkv: rows split in three, the down projection shared.
        guard case .lowRank(let upK, _, _) = m.updates["transformer_blocks.0.attn.add_k_proj"]!,
              case .lowRank(let full, _, _) = updates["diffusion_model.double_blocks.0.txt_attn.qkv"]! else {
            Issue.record("expected low rank")
            return
        }
        #expect(abs(upK - full[4 ..< 8]).max().item(Float.self) == 0)
        // adaLN: halves swapped.
        guard case .lowRank(let swapped, _, _) = m.updates["norm_out.linear"]!,
              case .lowRank(let original, _, _) = updates["final_layer.adaLN_modulation.1"]! else { return }
        #expect(abs(swapped[0 ..< 4] - original[4 ..< 8]).max().item(Float.self) == 0)
    }

    @Test func diffusersKeysRenameToFluxKit() {
        #expect(Flux2Weights.fluxKitKey("transformer_blocks.2.attn.to_out.0.weight") == "transformer_blocks.2.attn.to_out.weight")
        #expect(Flux2Weights.fluxKitKey("time_guidance_embed.timestep_embedder.linear_2.bias") == "time_guidance_embed.linear_2.bias")
        #expect(Flux2Weights.fluxKitKey("single_transformer_blocks.0.attn.to_out.weight") == "single_transformer_blocks.0.attn.to_out.weight")
    }
}

// MARK: - Real weights

enum KleinFixture {
    static let directory = Flux2Weights.defaultDirectory
    static var available: Bool { FileManager.default.fileExists(atPath: directory.appendingPathComponent("transformer").path) }
    static let owl = URL(fileURLWithPath: "/Library/User Pictures/Animals/Owl.heic")

    nonisolated(unsafe) private static var cached: (Flux2Model, ModelPair)?
    private static let lock = NSLock()

    /// The distilled mflux export, adapter absent (target = base) — plumbing only.
    static func pair(longSide: Int = 256) throws -> (Flux2Model, ModelPair) {
        try lock.withLock {
            if let cached, cached.0.longSide == longSide { return cached }
            let store = try TensorStore(componentDir: directory.appendingPathComponent("transformer"))
            let s = AdapterSwitch()
            let model = Flux2Model(directory: directory, transformer: try Flux2Transformer(store: store), adapterSwitch: s,
                                   longSide: longSide, transformerSource: "mflux")
            let pair = ModelPair(target: Flux2KleinBackend(identifier: "t", model: model, adapterOn: false, maxBatch: 2),
                                 base: Flux2KleinBackend(identifier: "b", model: model, adapterOn: false, maxBatch: 2))
            cached = (model, pair)
            return (model, pair)
        }
    }

    static func reference() throws -> ReferenceImage {
        FileManager.default.fileExists(atPath: owl.path) ? try ReferenceImage.load(url: owl) : Fixtures.image(width: 256, height: 256)
    }
}

enum Fixtures {
    static func image(width: Int, height: Int) -> ReferenceImage {
        var px = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                px[i] = UInt8(255 * x / width)
                px[i + 1] = UInt8(255 * y / height)
                px[i + 2] = UInt8(128 + 100 * sin(Double(x + y) / 20))
            }
        }
        let cg = ImageWriter.image(rgba: px, width: width, height: height)!
        return ReferenceImage(image: cg, name: "gradient")
    }
}

@Suite(.serialized, .enabled(if: KleinFixture.available, "Klein weights not on disk"))
struct Flux2WeightsTests {
    /// The hook wraps every linear; with the switch off the output is bitwise the base's.
    @Test func linearHookOffIsBitwiseTheBase() throws {
        let dir = KleinFixture.directory.appendingPathComponent("transformer")
        let plain = try Flux2Transformer(componentDir: dir)
        let s = AdapterSwitch()
        var wrapped = 0
        let hooked = try Flux2Transformer(componentDir: dir) { prefix, layer in
            guard prefix.hasSuffix("attn.to_q") || prefix.hasSuffix("to_qkv_mlp_proj") else { return layer }
            wrapped += 1
            let (o, i) = layer.logicalShape
            return DeltaLinear(base: layer, update: .lowRank(up: randn([o, 4], 7) * 0.01, down: randn([4, i], 8), scale: 1), switch: s)
        }
        #expect(wrapped == 25, "5 double + 20 single blocks")
        let x = randn([1, 64, 128], 11), enc = randn([1, 32, 7680], 12)
        let img = Flux2Pipeline.latentIDs(packedH: 8, packedW: 8), txt = Flux2Pipeline.textIDs(count: 32)
        let a = plain(hidden: x, encoder: enc, timestep: 500, imgIDs: img, txtIDs: txt)
        let b = hooked(hidden: x, encoder: enc, timestep: 500, imgIDs: img, txtIDs: txt)
        #expect(abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0)
        s.enabled = true
        let c = hooked(hidden: x, encoder: enc, timestep: 500, imgIDs: img, txtIDs: txt)
        #expect(abs(a.asType(.float32) - c.asType(.float32)).max().item(Float.self) > 0)
    }

    /// VAE encode → decode reproduces the reference (test-only decode of the reference's own latent).
    @Test func encoderRoundTripsTheReference() throws {
        let (model, _) = try KleinFixture.pair()
        let image = try KleinFixture.reference()
        let e = try model.encode(image)
        #expect(e.shape == [32, 32, 32] || e.shape[0] == 32)
        #expect(e.grid == GridShape(rows: e.shape[1], cols: e.shape[2]))
        let mean = e.latent.mean().item(Float.self), sd = sqrt(e.latent.variance()).item(Float.self)
        #expect(abs(mean) < 0.5 && sd > 0.5 && sd < 2, "BN-normalized latents ≈ N(0, 1): mean \(mean), sd \(sd)")
        let decoder = try Flux2VAEDecoder(componentDir: KleinFixture.directory.appendingPathComponent("vae"))
        let packed = Flux2Packing.pack(e.latent.expandedDimensions(axis: 0)).transposed(0, 2, 1)
            .reshaped(1, 128, e.shape[1] / 2, e.shape[2] / 2)
        let out = decoder.decodePacked(packed).asType(.float32)          // (1, H, W, 3) in [-1, 1]
        let (w, h) = model.workingSize(for: image)
        let ref = (MLXArray(image.rgbPlanar(width: w, height: h), [3, h, w]).transposed(1, 2, 0) * 2 - 1).expandedDimensions(axis: 0)
        let mse = ((out - ref) / 2).square().mean().item(Float.self)
        let psnr = 10 * log10(1 / Double(max(mse, 1e-12)))
        #expect(psnr >= 24, "round-trip PSNR \(psnr) dB")
    }

    /// ε̂ = x_t + (1 − σ)·v̂ is an ε-prediction: x̂₀ beats the noisy input at every level. (The
    /// distilled transformer only improves on it modestly below σ = 0.5 — levels it never
    /// trained on — which is why real runs need the base transformer.)
    @Test func epsilonPredictionDenoises() throws {
        let (model, pair) = try KleinFixture.pair()
        let e = try model.encode(try KleinFixture.reference())
        let cond = try pair.target.condition(prompt: "")
        for lambda: Float in [-2, 0, 2] {
            let (xt, alpha, sigma) = CollapseFieldStage.noised(pair.target, x0: e.latent, logSNR: lambda, draws: 2, schedule: .research)
            let eps = pair.target.predictEpsilon(xt, logSNR: MLXArray([lambda, lambda]), conditions: [cond])
            let x0 = (xt - sigma * eps) / alpha
            let ref = e.latent.expandedDimensions(axis: 0)
            let err = (x0 - ref).square().mean().item(Float.self)
            let naive = (xt / alpha - ref).square().mean().item(Float.self)
            print(String(format: "Klein (mflux) λ = %+.0f σ = %.2f: x̂₀ error %.4f vs noisy input %.4f (ratio %.2f)",
                         lambda, sigma, err, naive, err / naive))
            #expect(err < naive, "λ = \(lambda): x̂₀ error \(err) vs naive \(naive)")
        }
    }

    /// Common random numbers through a real network: base vs base reads exactly zero.
    @Test func baseVersusBaseIsExactlyZero() throws {
        let (model, pair) = try KleinFixture.pair()
        let e = try model.encode(try KleinFixture.reference())
        let field = try CollapseFieldStage(estimator: HutchinsonReverse(probes: 1), draws: 2)
            .measure(models: pair, reference: e, prompt: ProbePrompt(text: "", source: "unconditional"), levels: [2],
                     adjacency: [], schedule: .research)
        #expect(field.collapse[0].allSatisfy { $0.allSatisfy { $0 == 0 } })
        #expect(field.retainedBase[0].contains { $0 != 0 }, "the base's own curvature is not zero")
    }

    /// Forward mode agrees with finite differences of ε̂ (bf16 activations: loose tolerance).
    @Test func jacobianVectorProductMatchesFiniteDifferences() throws {
        let (model, pair) = try KleinFixture.pair(longSide: 128)
        let e = try model.encode(try KleinFixture.reference())
        let cond = try pair.target.condition(prompt: "")
        let lambda: Float = 2
        let (xt, _, _) = CollapseFieldStage.noised(pair.target, x0: e.latent, logSNR: lambda, draws: 1, schedule: .research)
        let v = randn(xt.shape, 21)
        let l = MLXArray([lambda])
        let f: ([MLXArray]) -> [MLXArray] = { [pair.target.predictEpsilon($0[0], logSNR: l, conditions: [cond])] }
        let jv = jvp(f, primals: [xt], tangents: [v]).1[0]
        let h: Float = 0.05
        let fd = (f([xt + h * v])[0] - f([xt - h * v])[0]) / (2 * h)
        let rel = sqrt((jv - fd).square().sum() / fd.square().sum()).item(Float.self)
        #expect(rel < 0.15, "‖Jv − FD‖/‖FD‖ = \(rel)")
        let back = vjp(f, primals: [xt], cotangents: [v]).1[0]
        #expect(abs((back * v).sum().item(Float.self) - (jv * v).sum().item(Float.self))
                / max(abs((jv * v).sum().item(Float.self)), 1e-3) < 0.1, "vᵀJv agrees in both modes")
    }
}

// MARK: - The undistilled base (FLUX.2-klein-base-4B), when downloaded

enum KleinBaseFixture {
    static var transformer: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huggingface/models/\(Flux2Weights.baseRepo)/transformer")
    }
    static var available: Bool {
        KleinFixture.available && FileManager.default.fileExists(atPath: transformer.appendingPathComponent("diffusion_pytorch_model.safetensors").path)
    }
}

@Suite(.serialized, .enabled(if: KleinBaseFixture.available, "FLUX.2-klein-base-4B transformer not downloaded"))
struct Flux2BaseTests {
    /// The renamed diffusers transformer denoises where the probe looks (σ ≤ 0.5) — and better
    /// than the distilled transformer does there.
    @Test func baseTransformerDenoisesAtLowNoise() async throws {
        let request = BackendRequest(backend: "flux2-klein", reference: try KleinFixture.reference(),
                                     options: ["long-side": ["256"], "max-batch": ["2"]])
        let pair = try await Flux2KleinFactory.make(request)
        #expect(pair.issues.isEmpty, "\(pair.issues)")
        #expect(pair.provenance["transformer-variant"] == "base")
        let e = try pair.target.encode(request.reference)
        let cond = try pair.target.condition(prompt: "")
        for lambda: Float in [-2, 0, 2] {
            let (xt, alpha, sigma) = CollapseFieldStage.noised(pair.target, x0: e.latent, logSNR: lambda, draws: 2, schedule: .research)
            let eps = pair.target.predictEpsilon(xt, logSNR: MLXArray([lambda, lambda]), conditions: [cond])
            let ref = e.latent.expandedDimensions(axis: 0)
            let err = ((xt - sigma * eps) / alpha - ref).square().mean().item(Float.self)
            let naive = (xt / alpha - ref).square().mean().item(Float.self)
            print(String(format: "Klein base λ = %+.0f σ = %.2f: x̂₀ error %.4f vs noisy input %.4f (ratio %.2f)",
                         lambda, sigma, err, naive, err / naive))
            #expect(err < naive)
        }
    }

    /// End to end through a local PEFT LoRA file: extraction, mapping, the delta applied, and
    /// specificity — a small random adapter is not memorization.
    @Test func randomAdapterIsAppliedButNotMemorization() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("scorpion-lora-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        var tensors: [SafetensorsWriter.Tensor] = []
        for block in 0..<5 {
            for proj in ["to_q", "to_k", "to_v"] {
                let base = "transformer.transformer_blocks.\(block).attn.\(proj)"
                let a = randn([4, 3072], UInt64(100 + block * 3)) / Float(3072).squareRoot()
                let b = randn([3072, 4], UInt64(200 + block * 3)) * 0.05
                tensors.append(.init(name: base + ".lora_A.weight", floats: a.asArray(Float.self), shape: [4, 3072]))
                tensors.append(.init(name: base + ".lora_B.weight", floats: b.asArray(Float.self), shape: [3072, 4]))
            }
        }
        let file = dir.appendingPathComponent("adapter_model.safetensors")
        try SafetensorsWriter.encode(tensors).write(to: file)
        try Data(#"{"r": 4, "lora_alpha": 4}"#.utf8).write(to: dir.appendingPathComponent("adapter_config.json"))

        let reference = try KleinFixture.reference()
        let request = BackendRequest(backend: "flux2-klein", model: file.path, reference: reference,
                                     options: ["long-side": ["256"], "max-batch": ["2"]])
        let pair = try await Flux2KleinFactory.make(request)
        #expect(pair.provenance["adapter-applied"] == "15 linear(s)", "\(pair.provenance)")
        #expect(pair.issues.isEmpty, "\(pair.issues)")
        #expect(pair.provenance["adapter-pin"]?.hasPrefix("sha256:") == true)

        var config = MemorizationConfiguration.quick
        config.maxLevels = 2
        config.draws = 2
        config.confirmationDraws = 2
        let result = try MemorizationProbe(configuration: config).run(
            models: pair, reference: try pair.target.encode(reference), prompt: ProbePrompt(text: "", source: "unconditional"),
            schedule: .research)
        print("random adapter: \(result.verdict.rawValue), regions \(result.regions.map { String(format: "%.2f@%.1f×", $0.meanCollapse, $0.snap) }), levels \(result.levels.map { String(format: "λ%g mean %.4f max %.3f", $0.logSNR, $0.meanCollapse, $0.maxCollapse) })")
        #expect(result.levels.contains { $0.maxCollapse != 0 }, "the adapter changes the function")
        #expect(result.verdict != .memorized)
    }
}
