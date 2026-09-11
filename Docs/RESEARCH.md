# Scorpion — research notes (memorization heatmaps)

## The question

> Has this model memorized regions of this image?

Scorpion started from a capability question: can the weights, given the right seed, produce a target likeness? That work is archived at git tag `archive/likeness-needle-v1`; §7 below summarizes what it established.

It is not the question that matters as evidence. "A model trained on similar images can produce something similar" is true of any general model. "These regions of *this* image were trained into the weights" is a factual, testable claim about what the model contains. Our own theory agrees: every image is reachable under a bijective sampler, so reachability can't discriminate. What discriminates is a *membership* contrast: what the weights pin, compared with a baseline that doesn't.

**Why the weights alone aren't enough.** A safetensors file carries no compute graph. Memorization is a property of the function the weights compute: its curvature at the image. So the probe has to evaluate the denoiser.

"Any model format" holds at the file layer. ScorpionKit reads safetensors, including over HTTP ranges, and decodes adapter conventions without knowing the architecture. Each model *family* then needs an executor that turns those tensors into a runnable model. The probe math is written once, against `LatentEncoder`/`PromptConditioner`/`Denoiser`.

## 1. Memorization as variance collapse

**What the paper measures.** Kim & Lee, *Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences* (arXiv 2605.26756):
- Memorization is a failure to generalize: the learned local intrinsic dimension falls below the true one (Ross et al., ICLR 2025).
- *Verbatim* memorization is **coordinate-wise variance collapse**: some coordinates (a template, a logo, a face) stop varying.

**Second-order Tweedie turns curvature into variance** (their Prop. 4.1). With J = ∂ε̂/∂x_λ:

    Var[x₀,ᵢ | x_λ] = (σ²/α²)·ρᵢ,   ρᵢ = 1 − σ·Jᵢᵢ

ρ is the **retained variance**: 1 means the coordinate is still free at this noise level, 0 means it is pinned. The identity uses only ε̂ = −σ∇log p_λ and x_λ = αx₀ + σε, so it holds for **any** (α, σ):
- variance-preserving models (α² + σ² = 1);
- flow-matching models (α + σ = 1). For flow models ε̂ = x_t + (1 − σ)·v̂ is exact algebra from the velocity prediction.

**Why a baseline is needed.** Curvature also fires on naturally low-dimensional content (flat backgrounds), so collapse is measured *against an underfitted baseline*:
- **C = ρ_base − ρ_target:** what the weights pin that the baseline leaves free. The paper's Δh_θ̃, with a less-trained checkpoint or the pre-fine-tune base.
- **C∅ = ρ_uncond − ρ_cond under the target:** what the prompt pins (Δh_∅). No second model is needed. It exposes conditional overfitting: a trigger word whose conditional distribution collapsed before the unconditional one did.

**How memorization happens.**
- The optimal denoiser for a finite training set sends every trajectory into a training point (Biroli et al., Nat. Commun. 2024).
- Networks reach that regime when trained past τ_mem ∝ n (Bonnaire et al., NeurIPS 2025).
- So memorization is unintentional whenever the effective n per condition is small:
  - duplicated images (Carlini et al. 2023; Somepalli et al. 2023);
  - few-image fine-tunes run for thousands of steps;
  - unique trigger tokens.
- Across a subject's images, the subject is the invariant, so identity fine-tunes memorize it *locally*: the paper's template-verbatim case.

## 2. The probe

Every statistic is fixed before the probe runs and recorded in the report.

**Where it is measured.** The probe evaluates at the *noised reference*, x_λ = α·x₀ + σ·ε, using keyed draws that every model shares. The paper uses generated samples instead; Scorpion never generates for display, and the reference is what's on trial.

**1. Level scout (base only).**
- Collapse only means something where the base leaves the image free: 0.5 ≤ ρ_base ≤ 1.1.
- Below that range every model pins the image. Above it, a base that doesn't know the content hesitates *between* contents, which reads as collapse for any model that merely learned it.
- The scout measures ρ_base alone, on its own draws.
- It keeps the contiguous run of levels where the base releases the most area. When that run is too long for the profile, the window starts where the base first releases most of the image, because a memory pins content from the base's natural variability down to its own width.
- The choice cannot depend on the target.

**2. Collapse field.**
- Per level and per draw: C and C∅ averaged over channels. The paper sums; the mean keeps ρ units for any latent width.
- diag J comes from Hutchinson probes, E[v ⊙ Jᵀv] over Rademacher v, one VJP per probe. The forward-mode E[v ⊙ Jv] is available too.
- Probe vectors and noise draws are keyed by (λ value, probe) and generated at the full batch shape before slicing, so chunking never changes a number.
- Every model and prompt sees the same vectors, so differences are paired: a model compared with itself reads exactly zero.

**3. Regions.**
- **Gate per position:** smoothed ρ_base and its forward spread must lie within [0.5, 1.1], and the raw ρ_base must be ≤ 2.2.
- **Sustain:** S(p) = max over adjacent gated level pairs of min(C̄(p,ℓ), C̄(p,ℓ+1)). A memory pins content over a band of noise levels; a base's hesitation occupies one narrow window.
- **Segment:** positions with S ≥ τ = 0.1 whose one-sided 95% t lower bound (built the same way) is above 0 form 4-connected clusters with a minimum area.
- **Mass:** a cluster's mass is ΣS.

**4. Null calibration.**
- The problem: a per-position threshold over thousands of positions fires somewhere by chance, and the firings are spatially correlated.
- So each region is tested as a whole. Negative controls go through the same steps, and p = (1 + #{controls with max cluster mass ≥ mass}) / (1 + N).
- The right controls are held-out images the user supplies: a subject's never-published photos are guaranteed non-members. Base-model latent samples are a weaker fallback, never decoded.
- With N = 19, the smallest attainable p is 0.05; the `evidence` profile requires N ≥ 19. Without controls, regions are reported as uncalibrated.

**5. Confirmation.**
- Collapse says the model pins a region. It doesn't say the region was pinned onto *this* image: an identity fine-tune pins any photo of the person onto its stored photos.
- So, on keyed draws **independent** of those that chose the region (sample splitting), compute snap = ‖x̂₀_base − x₀‖² / ‖x̂₀_target − x₀‖², as a geometric mean of per-cell ratios over the region, at its peak level.
- The membership statistic is the band snap over all candidate regions.

**6. Verdict (rule-based, uncalibrated).**

| Verdict | Condition |
|---|---|
| `inconclusive` | The test was invalid: no released levels, a backend issue, too few controls, or a baseline that is freer than the target almost everywhere |
| `memorized` | A counted region with snap ≥ 4 |
| `memorized-other-content` | A counted region pinned onto something else |
| `no-evidence` | Nothing the controls don't also show. Not evidence of non-use. |

**Heatmap layers.**

| Layer | Contents |
|---|---|
| `collapse` | Sustained collapse inside counted regions; the primary layer |
| `collapse-raw` | Sustained collapse everywhere |
| `prompt-collapse` | C∅, for conditional prompts |
| `score-difference` | The forward-only surrogate (ε̂_T − ε̂_B)² — Wen et al.'s detector. It conflates learning with memorization. |

The colour scale is absolute and never normalized per image, so a model that memorized nothing shows nothing.

**Optional: seed-endpoint basin.**
- Invert the reference to its seed z*, move z*'s segment under the region by β with pCN (√(1−β²)·z* + β·ξ stays N(0, I)), resample the surroundings, and count returns r(β).
- r(0) is reachability, r(1) the region's mass under random seeds, and β½ the attractor's radius.
- It lives in `Memorization/SeedBasin/`, depends on the core result, and is never needed by it.

## 3. Validation on toy worlds with known memorization

`MemoryScenario` plants memorized regions in a Gaussian-mixture world built around the reference. Each region holds the subject as a matched-variance kernel density over n training images:

    components at μ̂ + c(xᵢ − μ̂), variance h² = (1 − κ)s², c = √(1 − h²/s²)

κ = 0 is learned (one generalized Gaussian); κ = 1 is the training images themselves. The total variance is the same at every κ, so only *how* content is held changes. Regions are separate mixture blocks, so the truth is exact: curvature and posterior variance are closed-form (`exactDiagHessian`, `exactPosterior`).

| Check | Result |
|---|---|
| Closed-form curvature vs autodiff; Prop. 4.1 vs mixture posterior | relative error < 10⁻³ |
| Hutchinson region means (K = 64) | within 0.02 of exact |
| Forward vs reverse Hutchinson (symmetric exact Jacobian) | agree to 10⁻⁴ |
| Chunking (maxBatch 3 vs whole) | identical estimates |
| Estimated vs exact collapse over the planted region, λ = 3, 4, 5 | within 0.03 |
| Stored region (κ = 1) | `memorized`, IoU ≥ 0.9, mean collapse ≥ 0.6, background < 0.1 |
| Intensity ordering (κ = 1, 0.6, 0 in one world) | stored > half-memorized; learned region not flagged |
| Dose-response κ = 0, 0.5, 0.9, 1 | region collapse strictly increasing; κ = 0 → none |
| Target = base | every layer exactly 0 |
| Held-out reference | never `memorized` |
| Trigger-gated memory | prompt collapse ≥ 0.3; ungated → < 0.02 |
| Learned novel content (base hesitates) | no region, although ungated single-level collapse > 0.1 |
| 19 controls, stored region | p = 0.05, `memorized`; learned content → `no-evidence` |
| Membership (5 members vs 5 held-out) | AUC ≥ 0.9 at κ = 1, chance at κ = 0 |
| Seed basin, stored region | β½ target > base + 0.2; r(1) matches brute-force μ |
| Go/no-go dry run (96 references) | GO: membership 1.00, IoU 1.00, Spearman 1.00, 0% false flags |

**What building it taught us** (each is now a test or a design rule):
1. **The baseline decides what collapse means.** A novel face the base has never seen reads as collapse at high noise, because the base hesitates between contents. Per-position release gating plus the two-level rule removes it; the unconditional branch (C∅) removes it more cleanly when a prompt gates the memory.
2. **One image is enough for memorization, and only for memorization.** Membership AUC is 1.0 when images are stored and chance when content is only learned: there is nothing image-specific to find.
3. **Held-out images of a memorized subject are pinned onto the stored ones.** The honest reading is `memorized-other-content`, not `memorized`.
4. **Region boundaries matter for snap.** Map smoothing widens regions by its radius. A ratio of summed distances was dominated by that fringe (77× → 2.5×); the geometric mean of per-cell ratios is robust to it (→ 22.5×).
5. **Memorization is band-limited in noise.** A pool pins content from the base's variability down to its own width. The paper's final-step choice suits point-like memories; a finite-width pool is invisible at t → 0.

## 4. FLUX.2 Klein executor

**Layout.**
- `ScorpionFlux2` wraps Frigate's FluxKit.
- Latents are the VAE's BN-normalized latents unpacked to [32, H/8, W/8]; the transformer sees 2×2-packed 128-channel tokens.
- Flow schedule: σ = sigmoid(−λ/2), α = 1 − σ, timestep = 1000σ.

**Adapters.**
- ScorpionKit's format-level extraction decodes the adapter under test (LoRA, LoHa, LoKr, DoRA-approx, diffs; Hub links or local files) to ΔW per module.
- `Flux2LoRAMapper` places ΔW on FluxKit's linears, which use mflux names:
  - diffusers names match except `attn.to_out.0` and `time_guidance_embed.timestep_embedder.*`, checked against both checkpoints' tensor names;
  - kohya underscores and PEFT prefixes are handled;
  - BFL-native fused qkv is split row-wise;
  - the final adaLN halves are swapped.
- `DeltaLinear` applies ΔW at run time through a `linearTransform` hook added to FluxKit's `TensorStore`. One transformer serves target (switch on) and base (switch off), so the base is exactly the base.

**Checks on the real 4B weights** (`ScorpionFlux2Tests`, run when the weights are on disk):

| Check | Result |
|---|---|
| Hook with the adapter switched off | bitwise identical to the unhooked transformer |
| Base vs base collapse field (λ = 2, VJP) | exactly 0 everywhere; the base's own ρ is non-zero |
| VAE encode→decode of the reference (test only) | ≥ 24 dB |
| Jv vs central finite differences of ε̂ (bf16) | relative error < 0.15; vᵀJv agrees between JVP and VJP |
| x̂₀ error / noisy-input error, distilled (mflux 4-bit) | 0.07 at σ = 0.73, 0.26 at σ = 0.5, 0.63 at σ = 0.27 |
| Same, base (diffusers bf16, renamed to FluxKit keys) | 0.05, 0.21, 0.56 |
| Base vs base on klein-base-4B, full CLI (`quick`, 256 px) | exactly 0, `no-evidence`, 104 evaluations in 98 s |
| Random rank-4 LoRA from a local PEFT file (15 linears) | applied with no issues; max collapse 0.015; `no-evidence` |

**Distilled vs base.** The distilled Klein is trained only on its 4-step path (σ ≈ 1, .96, .88, .72) and denoises poorly at σ ≤ 0.5, where the release gate lives. Runs on it are therefore `inconclusive`, and the default transformer is FLUX.2-klein-base-4B, which Klein LoRAs are trained on. It loads from BFL's diffusers export (bf16, about 7.8 GB).

## 5. Limitations

- **Uncalibrated.** τ, the snap threshold and the verdict rules are uncalibrated until the real-model experiment (Docs/EXPERIMENT.md) runs on LoRAs with known training sets. Until then every report says `calibrated: false`.
- **Verbatim memorization only.** A well-generalized concept leaves no localized collapse (the paper's own limit). Absence of memorization is not evidence of non-use.
- **Triggers.** A memory gated behind a trigger the probe doesn't use can stay hidden (Kowalczuk et al., *Finding Dori*). Reports state the prompt.
- **Real networks are not optimal denoisers.** Tweedie's identities are exact for exact scores. For trained networks, ρ is the network's own curvature, which is the quantity the paper validates.
- **One executor so far.** Other families need executors. Pickle checkpoints are refused by design; GGUF payloads are not decoded.

## 6. Boundaries

- **Reference-directed.** The probe asks whether *this* image is stored, and never enumerates what a model stored; that would be training-data extraction.
- **Decode-free.** Latents never become images; the heatmap is drawn over the user's own reference. The one exception is a test-only VAE round-trip of the reference's own latent.

## 7. Archive: what the likeness work established

The face-seed needle, band likelihood (Δbits) and weight-space screening are at `archive/likeness-needle-v1`. Results that still inform the design:
- **Reachability is always "yes"; probability is what differs.** The noise → image map is a bijection, so every image has a pre-image.
- **Seed permutations as common random numbers** cut Monte Carlo variance about 6.7× (reused here: shared draws and probes).
- **Likelihood gain measures what a fine-tune added, not what it stored.** One learned identity gained 2 bits in one toy world and 862 in another, depending on what the base already knew. That is why memorization, not likelihood, is the evidence.
- **Few-step samplers over-produce small components.** Sampler choice is part of any seed-based claim, which is why the seed basin reports its sampler.

## References

- Kim & Lee, *Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences*, 2026 — https://arxiv.org/html/2605.26756v1
- Ross et al., *A Geometric Framework for Understanding Memorization in Generative Models*, ICLR 2025
- Kamkari et al., *A Geometric View of Data Complexity* (FLIPD), NeurIPS 2024
- Wen et al., *Detecting, Explaining, and Mitigating Memorization in Diffusion Models*, ICLR 2024 — https://openreview.net/forum?id=84n3UwkH7b
- Biroli, Bonnaire, de Bortoli, Mézard, *Dynamical Regimes of Diffusion Models*, Nature Communications 2024 — https://arxiv.org/abs/2402.18491
- Bonnaire, Urfin, Biroli, Mézard, *Why Diffusion Models Don't Memorize*, NeurIPS 2025 — https://arxiv.org/abs/2505.17638
- Jeon et al., *Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes*, ICML 2025 — https://arxiv.org/abs/2412.04140
- Zhai et al., *Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy*, NeurIPS 2024 — https://arxiv.org/abs/2405.14800
- Kowalczuk et al., *Finding Dori: Memorization in Text-to-Image Diffusion Models Is Less Local Than Assumed*, 2025 — https://arxiv.org/abs/2507.16880
- Asthana & Belagiannis, *Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability*, ICLR 2026 — https://arxiv.org/abs/2601.20642
- Carlini et al., *Extracting Training Data from Diffusion Models*, USENIX Security 2023
- Somepalli et al., *Diffusion Art or Digital Forgery?* (CVPR 2023) and *Understanding and Mitigating Copying in Diffusion Models* (NeurIPS 2023)
- Webster, *A Reproducible Extraction of Training Images from Diffusion Models*, 2023
- Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021 — https://arxiv.org/abs/2011.13456
- Hutchinson, *A stochastic estimator of the trace of the influence matrix*, 1989; Bekas et al., *An estimator for the diagonal of a matrix*, 2007
- Cotter, Roberts, Stuart, White, *MCMC methods for functions* (pCN), Statistical Science 2013
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023; Liu et al., *Rectified Flow*, ICLR 2023
