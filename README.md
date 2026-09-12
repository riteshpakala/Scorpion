# Scorpion

Has a diffusion model **memorized** regions of your image? Give Scorpion a model and a reference image. It measures where the model's weights pin that image more tightly than their baseline does, and draws the answer as a **heatmap over your own image**. Nothing is generated for display, and no latent is ever decoded.

Scorpion answers "these regions of the image were trained into the model". It does not answer "the model can produce something similar". The first is a claim about what the weights contain, which is the defensible form of the question. The second is a capability claim, which any sufficiently general model satisfies.

It ships as a Swift package:

| Product | What it is |
|---|---|
| `ScorpionKit` | The model-agnostic library: weight extraction (safetensors over HTTP ranges, adapter decoding), the memorization probe, heatmaps, reports, and the analytic toy backend |
| `ScorpionFlux2` | The first real executor: FLUX.2 Klein 4B through Frigate's FluxKit |
| `scorpion` | Command-line tool |
| `ScorpionApp` | macOS app, launched from `main.swift` via `NSApplication` |

## Requirements

- macOS 14+ on Apple Silicon
- Swift 6.3+ (Xcode 26) with the Metal toolchain (`xcrun metal` must work)

MLX and FluxKit come from [Frigate](https://github.com/rao-studios/Frigate), pinned by revision. MLX still loads a small precompiled library at startup, which `swift build` can't produce, so build it once after each build:

```sh
swift build
scripts/build-metallib.sh          # builds mlx.metallib and places it next to the binaries
```

Re-run the script after `swift build -c release` or after cleaning `.build`. For tests: `swift build --build-tests && scripts/build-metallib.sh && swift test --skip-build --no-parallel`. Run the suites serially: MLX Swift's eval lock and its compiled-function lock can deadlock when two threads evaluate models at once. For the same reason, one process runs one probe at a time. The Klein tests run when the weights are on disk and skip otherwise.

## Usage

```sh
# The toy backend plants memorization in known regions of your image (x,y,w,h in [0,1] :κ),
# so you can see what a positive looks like. Writes the heatmap figure and the JSON report.
.build/debug/scorpion memorization --image ~/ref.jpg --plant 0.2,0.2,0.3,0.3:1 --overlay heatmap.png --json report.json

# A FLUX.2 Klein LoRA (Hugging Face / GitHub link or a local .safetensors) against your image,
# with negative controls: a folder of images known NOT to be in its training set.
.build/debug/scorpion memorization --backend flux2-klein --model https://huggingface.co/owner/klein-lora \
    --image ~/ref.jpg --controls ~/never-published --profile evidence --overlay heatmap.png --json report.json

# Optional: the seed-endpoint basin on the strongest region (slow on real models).
.build/debug/scorpion memorization ... --basin

# Header-only look at a model: formats, adapter type, structural family. No weights fetched.
.build/debug/scorpion inspect --model https://huggingface.co/nerijs/pixel-art-xl

# Validation where the answer is known: sweep memorization strength κ on analytic worlds.
.build/debug/scorpion research memory --image ~/ref.jpg
.build/debug/scorpion experiment run --manifest Docs/examples/toy-memory.json
```

**Options**

| Option | Meaning |
|---|---|
| `--profile` | `quick` \| `standard` \| `evidence` (needs ≥ 19 controls) \| `toy` |
| `--prompt` | Condition on this prompt. The default is the model's trigger word, else unconditional. |
| `--controls DIR` | Held-out images that calibrate regions to p-values |
| `--base-controls N` | Base-model latent samples as weaker controls |
| `--estimator` | `hutchinson-reverse` (default) or `hutchinson-forward` |
| `--layer` | Which heatmap to draw: `collapse`, `collapse-raw`, `prompt-collapse` or `score-difference` |

**Klein options** (`--option key=value`)

| Option | Meaning |
|---|---|
| `transformer` | `base` (default: downloads FLUX.2-klein-base-4B once, about 7.8 GB), `mflux` (the distilled 4-bit export; plumbing only) or a path to a diffusers transformer |
| `base-dir` | The mflux export holding the text encoder, VAE and tokenizer (default `~/Documents/huggingface/models/mlx-community/flux2-klein-4b-4bit`, or `SCORPION_FLUX2_DIR`) |
| `quantize` | 4 or 8 |
| `long-side` | Default 512 |
| `max-batch` | Default 4 |

**Seeds.** Seeds come from a keyed schedule. The key is `SCORPION_SEED_KEY`, or else a per-install random key in `~/Library/Application Support/Scorpion/seed.key`. Reports record the key's hash, never the key.

**Private repos.** For gated or private repos, set `HF_TOKEN` / `GITHUB_TOKEN`. Each token is only sent to its own host.

**The app.** Run `scripts/run-app.sh`. It builds the app, puts `mlx.metallib` next to it when that's missing, and launches it. Add `--release` for faster probes on real models, or `--detach` to return to the shell (logs go to `.build/ScorpionApp.log`); `--help` lists the rest. Pick a backend, drop an image, optionally choose a controls folder, and press **Test**. The heatmap is drawn over the reference, with a layer picker, an opacity slider, the regions, the noise levels, the verdict and provenance, and PNG/JSON export.

## How it works

Memorization is **coordinate-wise variance collapse**: the model stops letting some pixels vary. Kim & Lee ([arXiv 2605.26756](https://arxiv.org/html/2605.26756v1)) read it as curvature against an underfitted baseline. By second-order Tweedie, each coordinate's *retained variance* at noise level λ is

    ρ = 1 − σ·diag(∂ε̂/∂x_λ)        1 = still free here, 0 = pinned

Scorpion measures ρ at the **noised reference** itself (x_λ = α·x₀ + σ·ε, keyed ε shared by every model), in six stages. Each stage is its own type in `Sources/ScorpionKit/Memorization/`:

| # | Stage | What it does |
|---|---|---|
| 1 | `LevelScout` | Picks the noise levels where the *base* leaves the image free. It looks at the base only, so the choice cannot depend on the target. |
| 2 | `CollapseFieldStage` | Computes per-position collapse **C = ρ_base − ρ_target**, and **C∅ = ρ_uncond − ρ_cond** for a conditional prompt. Uses a Hutchinson estimate of diag J, one VJP per probe, paired across models. |
| 3 | `RegionDetector` | Gates each position on where the base releases it. Keeps collapse that is *sustained* over two adjacent levels with a t lower bound above zero, then segments it into 4-connected regions. |
| 4 | `NullCalibration` | Runs the same steps on negative controls. A region's p-value = (1 + #controls with max cluster mass ≥ its mass) / (1 + N). |
| 5 | `ConfirmationStage` | Uses **independent** draws: does the target return the region to the reference itself? Snap = base distance / target distance, a geometric mean over cells. |
| 6 | `VerdictRule` | `memorized` · `memorized-other-content` · `no-evidence` · `inconclusive`. The last means the test itself was invalid: no released levels, an adapter trained on another base, too few controls, and so on. |

**The heatmap.**
- It is the sustained collapse on a **fixed absolute scale**: transparent below τ = 0.1, fully dark at 1 (pinned).
- It is never normalized per image. A per-image stretch would always show a hot spot, even on a model that memorized nothing.
- It is drawn over a washed-out copy of *your* image, beside the untouched original, with numbered region outlines.
- The JSON carries the raw grid, every layer, the regions, their p-values and snaps, the levels, and provenance. Provenance includes content hashes, model pins and the schedule id.

**The optional seed-endpoint basin** (`Memorization/SeedBasin/`) builds on the core result; the core never depends on it.
- It inverts the reference to its seed z*.
- It moves z*'s segment under the strongest region by β (preconditioned Crank–Nicolson) and resamples the rest.
- It counts how often the model comes back. r(0) is reachability, r(1) is the chance a random seed lands there, and β½ is the attractor's radius in seed space.

## Example (toy backend, macOS sample `Owl.heic`)

The default plants are a stored region (κ = 1) and a partly memorized one (κ = 0.8):

```
Scorpion 0.2.0 · MEMORIZED (rule-based, uncalibrated)
  #   area    mean   peak   λ    mass     p        significance     snap    bounds (px)
  1    10.7%   0.79   0.79  5       87.3  —        uncalibrated       73.7  80,96 176×160
  2     7.6%   0.27   0.95  2       21.3  —        uncalibrated       98.0  288,256 144×160
```

With `--profile evidence --base-controls 19`, both regions are significant at p = 0.05, the smallest p that 19 controls allow. The verdict stays `MEMORIZED`, with snap 22.5× on independent draws.

The go/no-go protocol on the toy (`experiment run`, 3 subjects × κ ∈ {0, 0.5, 0.9, 1}):
- membership AUC 1.00;
- localization IoU 1.00;
- dose-response Spearman 1.00;
- 0% false flags at κ = 0;
- verdict **GO**.

## FLUX.2 Klein

The executor runs Klein's transformer with the adapter under test applied at run time, through a linear hook in FluxKit.
- One transformer serves both sides: the adapter on is the target, off is the base.
- The base is therefore *exactly* the base, and every other weight is shared bit for bit.
- Latents are the VAE's normalized latents at H/8 × W/8, so the heatmap has 64 × 64 cells at 512².

**Checked on the real 4B weights:**

| Check | Result |
|---|---|
| Hook with the adapter off | Output is bitwise identical to the unhooked transformer |
| Base vs base | Every collapse map is exactly 0 |
| VAE encode→decode of the reference | ≥ 24 dB (test only) |
| Jv | Matches finite differences of ε̂ |
| Base vs base on klein-base-4B (full CLI) | Maps exactly 0, `no-evidence` |
| Random rank-4 LoRA from a local file | Applied to 15 linears with no issues; max collapse 0.015; `no-evidence` |

**Use the base transformer.** Klein LoRAs are trained on **FLUX.2-klein-base-4B**. The distilled Klein only ever runs its few sampling steps (σ ≈ 1 … 0.72), and its denoising degrades at the σ ≤ 0.5 levels the probe needs. Measured x̂₀ error relative to the noisy input: 0.07 at σ = 0.73, 0.26 at σ = 0.5, 0.63 at σ = 0.27 (the base: 0.05, 0.21, 0.56). So runs on the distilled transformer are marked `inconclusive`.

**Cost at 512².** One forward pass is about 0.8 s on an M4 Max, and a VJP about 3× that. Approximate wall times:

| Profile | Time |
|---|---|
| `quick` | a few minutes |
| `standard` | about 20 min |
| `evidence` | adds a standard run per control (cacheable per model and prompt) |

## Package layout

```
Sources/ScorpionKit/
  Scorpion.swift        façade: inspect · extractAdapter · detectMemorization
  Backend/              LatentEncoder / PromptConditioner / Denoiser, ModelPair, BackendRegistry, DeterministicSampler
  Memorization/         the six stages, estimators (Hutchinson reverse/forward, exact), profiles, results
    SeedBasin/          the optional seed-endpoint basin
  Heatmap/              heatmap data, colormap, renderer, PNG writer
  Report/               MemorizationReport (scorpion.memorization.v1)
  Toy/                  analytic GMM backend, planted-memory worlds, κ sweep, go/no-go runner
  Source/ Fetch/ Formats/ Weights/ Seeds/   link resolution, range fetching, safetensors/GGUF, adapter decoding, keyed seeds
Sources/ScorpionFlux2/  Klein backend, VAE/packing, LoRA mapper, DeltaLinear, diffusers weights, registration
Sources/scorpion/       CLI (composition root: registers executors)
Sources/ScorpionApp/    main.swift → NSApplication, AppKit menu, SwiftUI views
```

## Limits

- **Uncalibrated.** The rule-based verdict and the τ and snap thresholds are uncalibrated until the real-model memory experiment ([Docs/EXPERIMENT.md](Docs/EXPERIMENT.md)) runs on LoRAs with known training sets.
- **Verbatim memorization only.** Collapse localizes verbatim memorization. A well-generalized concept leaves no localized collapse (the paper's own limit). Absence of memorization is not evidence of non-use.
- **One executor.** Other model families need their own executor behind the same protocols.

## Archive

Face segmentation, the face-seed needle, band-likelihood research and the CLIP likeness scan were retired when Scorpion refocused on memorization. They live at git tag `archive/likeness-needle-v1`.

## License

GPL-3.0 — see [LICENSE](LICENSE).
