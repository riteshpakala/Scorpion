# Scorpion

Deepfake defense for model repositories. Give Scorpion a link to a model (Hugging Face or GitHub) and a reference image; it estimates how likely that model is to reproduce the reference's likeness — **without downloading the whole model and without generating images**.

It ships as a Swift package with three products:

| Product | What it is |
|---|---|
| `ScorpionKit` | The library: partial fetching, weight screening, reference analysis, seed permutations, probes, scoring |
| `scorpion` | Command-line tool |
| `ScorpionApp` | macOS app, launched from `main.swift` via `NSApplication` |

**Status: Phase 1 (model-agnostic).** Nothing in the code knows any specific architecture — only file formats, adapter conventions, linear algebra and text. See [Docs/RESEARCH.md](Docs/RESEARCH.md) for the theory and what Phase 1 can and cannot claim.

## Requirements

- macOS 14+ on Apple Silicon
- Swift 6.3+ (Xcode 26) with the Metal toolchain (`xcrun metal` must work)

MLX comes from [Frigate](https://github.com/rao-studios/Frigate) (pinned revision), which compiles MLX's Metal kernels just-in-time. MLX still loads a small precompiled library at startup, which `swift build` can't produce, so build it once after building:

```sh
swift build
scripts/build-metallib.sh          # builds mlx.metallib and places it next to the binaries
```

Re-run the script after `swift build -c release` or after cleaning `.build`. For tests: `swift build --build-tests && scripts/build-metallib.sh && swift test --skip-build`.

## Usage

```sh
# Header-only look at a model: formats, adapter type, structural family, fetch plan. No weights fetched.
.build/debug/scorpion inspect --model https://huggingface.co/nerijs/pixel-art-xl

# Describe a reference: content ID, faces, CLIP descriptors, seed permutations.
.build/debug/scorpion describe --image ~/reference.jpg

# Scan a model against a reference (JSON report with --json path, or - for stdout).
.build/debug/scorpion scan --model https://huggingface.co/owner/some-lora --image ~/reference.jpg --json report.json

# Validate the seed/band/region theory on analytic models built around the reference.
.build/debug/scorpion research gmm --image ~/reference.jpg

# Face-seed needle: segment the face, find the seed segment that makes it, and estimate how
# likely a random seed is to produce it (repeat --image for more photos of the same person).
.build/debug/scorpion research needle --image ~/me-1.jpg --image ~/me-2.jpg --controls ~/controls
.build/debug/scorpion describe --image ~/me-1.jpg --save-mask face-mask.png

# Memory pools: is your photo — or your face — stored in the weights? Learned vs stored,
# member vs held out, on analytic models with known memorization (repeat --image to add photos).
.build/debug/scorpion research memory --image ~/me-1.jpg

# The go/no-go experiment protocols (synthetic dry runs until Phase 2 executors exist).
.build/debug/scorpion experiment run --manifest Docs/examples/toy-experiment.json
.build/debug/scorpion experiment run --manifest Docs/examples/toy-memory.json

# Fingerprint bank and calibration.
.build/debug/scorpion bank add --model <link> --label identity
.build/debug/scorpion eval --manifest pairs.json --output calibration.json
.build/debug/scorpion scan ... --calibration calibration.json
```

Useful options: `--budget 256MB` (max tensor bytes to fetch), `--seeds 64`, `--band -4,2` (log-SNR range), `--face N`, `--no-clip`. Gated or private repos: set `HF_TOKEN` / `GITHUB_TOKEN` (each is only ever sent to its own hosts).

Seeds come from a keyed schedule. The key is `SCORPION_SEED_KEY` (hex or text), or else a per-install random key in `~/Library/Application Support/Scorpion/seed.key`. Reports record the key's hash, never the key. Research commands use a fixed public key so their numbers reproduce anywhere.

Supported links: `huggingface.co/{owner}/{name}[/tree|blob|resolve/{rev}/{path}]`, `github.com/{owner}/{repo}[/tree|blob/{ref}/{path}]`, GitHub release tags and assets, `raw.githubusercontent.com` and `media.githubusercontent.com` (Git LFS).

The app: `swift run ScorpionApp` (or `.build/debug/ScorpionApp` after `scripts/build-metallib.sh`). Paste a link, drop an image, click a face if there are several, and scan. **Run theory harness** shows the analytic validation for the loaded image.

### Example

A 170.5 MB SDXL LoRA (`nerijs/pixel-art-xl`) scanned against a photo of an owl:

```
Scorpion 0.1.0 · likeness harm score: 12/100 (provisional, uncalibrated)
Fetched    29 MB over 142 requests (17.0% of 170.5 MB weights)
Components
  metadata-match       0.02  no descriptor matched the model's metadata
  person-adapter       0.25  no person/likeness indicators in metadata
  spectral-narrowness  0.64  140 conditioning modules, mean effective rank 4.4, concentration 0.60
Unit       pixel-art-xl.safetensors · lora · 140/722 modules
           conditioning heuristic: read-only-width dims [2048] · 140 candidates
```

The conditioning heuristic found SDXL's 70 cross-attention layers (K and V, input width 2048) from tensor shapes alone, and only those matrices were fetched. `inspect` on the same model reads the headers with 318 KB.

## How it works (Phase 1)

1. **Resolve and pin** the link to a commit sha and list the files with their sizes.
2. **Read headers only.** A safetensors file starts with a JSON header of every tensor's name, dtype, shape and byte offsets. Two HTTP Range requests inventory a multi-GB file. GGUF headers are read the same way; pickle files (`.ckpt/.pt/.bin`) are flagged and **never deserialized**.
3. **Decode adapter conventions** that are shared across architectures: kohya/PEFT LoRA, LoHa, LoKr, DoRA, LyCORIS diffs, textual inversion, or a full model. Full models are diffed against the base named in their model card when the tensors line up.
4. **Find the conditioning readers** — the matrices that read the prompt embedding — from structure alone. Within a component, they read a width that no module writes, and they come in K/V sibling pairs.
5. **Fetch only those byte ranges**, within the budget, stratified over depth.
6. **Spectral screening.** For each update ΔW: the singular values (exact via QR for low-rank adapters), spectral entropy H = −Σ pᵢ log pᵢ, effective rank e^H, concentration σ₁²/Σσ², and the u₁ fingerprint. Low entropy means a few directions carry the change, which is the signature of a narrow learned concept such as one identity.
7. **Analyze the reference.** It gets a SHA-256 content ID, Vision face regions (soft masks with eyes, nose and mouth weighted), a CLIP zero-shot description, and visual statistics (luminance entropy, spectral slope, pHash).
8. **Seed permutations.** Noise and noise levels come from a keyed, fixed seed schedule (HMAC-SHA256), so they are identical for every reference, control and model: common random numbers everywhere. Each permutation fixes one noise draw ε, one log-SNR λ (stratified over the band) and one prompt from the reference's description (plus the model's trigger words), with a descriptor-swapped null prompt.
9. **Match and score.** Descriptors are matched against training tags, triggers and the model card. The fingerprint bank gives a nearest-neighbor vote. Components are fused in logit space, and reports are provisional until `scorpion eval` calibrates them.

The reference-specific likelihood probe (Δbits, face-region split, seed inversion) is implemented against the `DiffusionBackend` protocol and validated on an analytic backend. Real model executors plug into the same protocol in Phase 2. The JSON report records this explicitly.

### What the score means

The score is a capability estimate, not proof of misuse. In Phase 1 on real models, the signals say whether a model is a narrow person/likeness adapter *consistent with* the reference's visible attributes, not that it reproduces this specific identity. Identity-level evidence needs the likelihood probe with a real executor (Phase 2). Until the fusion is calibrated on labeled (model, reference, match) pairs, numbers are provisional, and every report says so.

## Theory harness results

`scorpion research gmm` builds Gaussian-mixture "models" around your reference, where likelihoods are known in closed form, and runs the real probe code against them. Output for the macOS sample image `Owl.heic`:

```
  variant          exact Δbits   estimated Δbits (90% CI)          region share   z*-atypicality (region)
  with-reference   +454.7        +441.4   [+426.6, +456.1]          100.0%         2.0
  with-other       -0.0          -0.0     [-0.0, +0.0]              —              28.7
  unchanged        +0.0          +0.0     [+0.0, +0.0]              —              28.7
Common random numbers: paired estimator variance is 14.8% of unpaired (6.7× fewer seeds for the same precision)
Seed contribution to the region, separable model: 100.0% of ∂x̂₀/∂ε mass inside a region covering 34.4%
Seed contribution to the region, coupled model:   46.3% of ∂x̂₀/∂ε mass inside a region covering 34.4% (concentration 1.6×)
```

## Face-seed needle

The face is where likeness harm concentrates. `research needle` turns "find the seed segment that makes this face, with the surrounding seed free" into a measurable rare-event probability:

1. **Segment** the face (Vision landmark hull plus person matte) and its eyes, nose and mouth.
2. **Measure the seed footprint:** which part of the seed decides the face, via autodiff through the sampler. Its share of the seed is the haystack reduction.
3. **Estimate the needle mass** μ = P(random seed → this face) by subset simulation. pCN moves search the face segment while the background seed is resampled freely; how often the face survives is the background independence ι.
4. **Report:**
   - −log₂ μ, the bits of luck an attacker needs;
   - Δbits against the base (what the weights save them);
   - 1 − (1−μ)^N, the chance of success within N seeds;
   - reachability of the inverted seed.

No face is ever decoded: the identity test is latent fidelity against a set of reference photos, calibrated so control faces don't count.

Validated where the answer is known, in analytic models whose true needle mass is the learned identity's mixture weight:
- estimates land within 0.3 decades of the truth down to 10⁻⁴;
- intervals cover the spread across keys;
- a separable face gives ι ≈ 1, and coupling lowers it.

What building it taught us (details in [Docs/RESEARCH.md §7](Docs/RESEARCH.md)):
- **The needle mass depends on the sampler.** Few steps over-produce small components.
- **Only models that put mass on the face reach it robustly.** Every image is reachable in principle, but the inverted seed under a model that doesn't know the face is a knife edge.
- **One photo is photo-level evidence.** Averaging over several photos of the person cuts false hits 30×.
- **Rare-event MCMC needs within-level step adaptation.**

The go/no-go experiment for real models is in [Docs/EXPERIMENT.md](Docs/EXPERIMENT.md).

## Memory pools

One photo is weak evidence of *identity*, but exactly the right evidence of *memorization*: models memorize photos, not people. Working backwards from Kim & Lee, *Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences* ([arXiv 2605.26756](https://arxiv.org/html/2605.26756v1)):
- **What memorization is:** coordinate-wise variance collapse. The model stops letting some pixels vary, which shows up as curvature.
- **How it happens:** trained long enough on few examples per condition, a model's denoiser drifts toward storing its training points. Duplicated photos, identity LoRAs and unique trigger words all qualify. The face is what stays constant across a person's photos, so the face is what gets pinned.
- **How it shapes later generations:** a stored photo becomes an attractor. Many seeds land on the same face, generations come back as near-copies, and prompts that never ask for the person can be pulled in.

`research memory` asks four decode-free questions of a reference:
1. **Is there a pool here?** Retained variance ρ = 1 − σ·diag(∂ε̂/∂x) at the noised reference, via Hutchinson VJPs. Collapse is ρ_base − ρ_target, counted only where the base leaves the face free.
2. **Is it yours?** Does the model denoise your noised photo back onto itself ("snap"), onto you, or onto someone else?
3. **Can a seed reach it?** Invert your photo to its seed endpoint, perturb the face segment, and count returns. r(0) is reachability, r(1) is the needle mass, and β½ is the pool's radius.
4. **Does it take over other generations?** Capture under trigger, descriptor and empty prompts, plus how often the person's generations recur as near-copies from different seeds.

The readout is a tier:
- `memorized-photo`: this photo, or a near copy, is stored;
- `identity-pool`: other photos of this face are stored;
- `nearby-pool`: a stored face near yours, not yours;
- `generalized-likeness`: learned, not stored, so it's the needle's case;
- `no-evidence`.

Validated where memorization strength κ is known (κ = 0 learned, κ = 1 stored). On the macOS sample `Owl.heic`:

```
κ      │ your photo was a training member                           │ your photo was held out
       │ tier               collapse   λ    snap    IoU   β½ T/B    │ tier               collapse  recur
0.00   │ generalized-likene   0.00     7     1.3   0.76   0.00/0.00 │ generalized-likene   0.00    0.00
0.50   │ memorized-photo      0.15     3    24.8   1.00   0.35/0.00 │ generalized-likene   0.04    0.00
0.90   │ memorized-photo      0.46     4    69.9   1.00   0.45/0.00 │ identity-pool        0.05    0.25
1.00   │ memorized-photo      0.68     5   124.2   1.00   0.50/0.00 │ identity-pool        0.00    0.56
```

Membership AUC was 0.97–1.00 once photos are stored and chance-level when the face is only learned. No look-alike or stranger was ever flagged. The probe only ever asks about a reference you provide: it does not list what a model memorized, which would amount to training-data extraction.

## Coding reference: Obscur in Frigate

Scorpion's direction follows Obscur's royalty-attribution logic in Frigate, run in the other direction. Obscur measures how much a generation *read* from registered images; Scorpion estimates how much a model's weights *could reproduce* a reference.

- [Frigate @ a19b127](https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2) — ObscurKit: per-corpus KV-bank adapters (Signets) and visual royalty attribution over FLUX.2 Klein
- [Frigate @ 1d3455e](https://github.com/rao-studios/Frigate/commit/1d3455e9b55e9ce01b093f7546c802cc88e7a040) — RMS-matched adapter branch and influence sweep

| Obscur (Frigate) | Scorpion |
|---|---|
| `ObscurEntry.id` = SHA-256 of image bytes | `ReferenceImage.id` = SHA-256 of reference bytes; roots every seed |
| Seeded projection init (`MLXRandom.key(seed)`) | Deterministic seed permutations → reproducible, auditable reports |
| `ObscurAttributionRecorder`: attention mass per entry / layer / step | Δbits per region / log-SNR sub-band / seed permutation |
| RMS-matched branch → dimensionless gate | Every score component dimensionless in [0, 1] |
| `ObscurStore` (manifest.json + bank.safetensors), `.topK` cosine | `FingerprintBank`, same layout, kNN over u₁ fingerprints |
| `ObscurAttributionReport` (versioned Codable) | `LikenessReport` (`scorpion.report.v2`) |
| "a causal-input record, not an influence certificate" | "a capability estimate, not proof of misuse" |

## Package layout

```
Sources/ScorpionKit/
  Scorpion.swift   façade (inspect / analyzeReference / scan / addToBank)
  Source/          link parsing, HF + GitHub resolution (commit-pinned)
  Fetch/           RangeFetcher (206-only, redirect-safe, streaming fallback), cache, ledger
  Formats/         safetensors, GGUF header, dtype decoding (f16/bf16/fp8)
  Weights/         inventory, adapter decoding, spectral stats, conditioning heuristic, metadata, analyzer
  Reference/       image loading, Vision faces, face/feature segmentation, visual stats, CLIP (MLX) + tokenizer, describer
  Seeds/           keyed seed schedule, seed engine (log-SNR bands, stratified λ, prompt sets)
  Probes/          DiffusionBackend, band likelihood, inversion, region sensitivity, analytic GMM backend,
                   deterministic sampler, seed footprint, subset simulation, face-seed needle,
                   curvature (Hutchinson, exact GMM), memory pools, experiment runner
  Bank/            fingerprint bank
  Report/          score fusion, calibration, metadata matching, LikenessReport
Sources/scorpion/  CLI
Sources/ScorpionApp/  main.swift → NSApplication, AppKit menu, SwiftUI views
scripts/build-metallib.sh
```

## Roadmap

- **Phase 2:** run the [go/no-go experiments](Docs/EXPERIMENT.md) with real `DiffusionBackend` executors — first FLUX.2 Klein via Frigate's FluxKit (ObscurKit's hooked attention sites can also expose reference-token attention mass), then other families. With an executor, every scan also gets the reference-specific Δbits, the face-region split, the inversion typicality and the memory-pool tier. The memory arm trains identity LoRAs to several checkpoints, so memorization onset can be measured against training time.
- A labeled calibration set of (model, reference, match) pairs built from consenting or synthetic identities.
- Populated fingerprint banks per structural family.

## License

GPL-3.0 — see [LICENSE](LICENSE).
