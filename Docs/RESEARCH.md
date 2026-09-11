# Scorpion — research notes (Phase 1, the face-seed needle and memory pools)

## The question

> In theory, can you reverse-engineer a seed to discover whether the weights can — given the right entropy — lead to a target generation? And can you isolate regions (like faces) in diffusion outputs and map them back to seed contributions?

**Short answer: yes in theory, with three corrections to the framing.** And it requires *evaluating* the network. The weights alone can't answer it, because a safetensors file contains named arrays and no compute graph.

## 1. Reverse-engineering the seed

**You recover the noise, not the integer seed.**
- The seed is an integer fed to a PRNG. What matters is the Gaussian noise tensor z it produces, which is the "entropy".
- Deterministic samplers (DDIM, DPM-Solver, flow-matching Euler, as used by SD, SDXL, Flux and SD3) are discretized ODEs, so the noise → image map is invertible. Integrating backward from a reference recovers its noise z* (DDIM inversion; null-text inversion handles classifier-free guidance).
- This works well enough in practice that Tree-Ring watermarks and Gaussian Shading read patterns planted in the initial noise back out of finished images.
- Integer seeds can only be recovered by searching a known candidate set. "Good Seed Makes a Good Crop" identifies the seed of a generated image at over 99.9% accuracy among candidates, which is a different question.

**Reachability is always "yes"; the question is probability.**
- The flow is a bijection, so every image, including your reference, has some z* that generates it, even under a model that never saw that face.
- The informative quantity is how much Gaussian noise mass lands on the target:

  log p_θ(x) = log N(z*; 0, I) + ∫ ∇·v_θ dt   (probability-flow ODE, Song et al. 2021)

- The "entropy required" is exactly the surprisal −log₂ p_θ(x) in bits. It has two readable parts:
  - Is z* a *typical* Gaussian draw?
  - How strongly does the flow contract noise onto the target?

**Raw likelihood is confounded by the image's own entropy.**
- Simple, low-detail images get high likelihood under every model (Nalisnick et al. 2019). So the score must be a likelihood ratio against a base model:

  Δbits = log₂ p_target(x) − log₂ p_base(x)

- This is roughly "how many bits of this reference the fine-tuned weights already hold." It is the defensible form of "X% likelihood".

## 2. A cheap, exact-in-the-limit estimator

For two diffusion models with exact denoisers, the pointwise log-likelihood difference is (Kong et al., *Information-Theoretic Diffusion*, ICLR 2023; Kingma et al., *VDM*, 2021):

    log p_T(x) − log p_B(x) = ½ ∫ E_ε[ ‖ε − ε̂_B(x_λ)‖² − ‖ε − ε̂_T(x_λ)‖² ] dλ,   x_λ = α(λ)·x + σ(λ)·ε

**The integral.**
- Each term costs **one denoiser call on a noised copy of the reference**. There is no sampling loop, no inversion, and no image is generated.
- The user's "range of pixels within a range of steps" is analytic: for seed ε at noise level λ, the model is handed exactly x_λ, and the only question is whether its prediction points back to the reference.

**Steps are a log-SNR band.**
- Step indices mean different noise levels in different schedulers.
- In λ = log(α²/σ²) the integrand is schedule-invariant (VDM), so a band of λ is comparable across model families. That is what makes the estimator model-agnostic.
- The default band [−4, 2] (SNR ≈ 0.018 … 7.4) is where content and identity form (cf. perception-prioritized weighting, Choi et al. 2022).

**Seed permutations are common random numbers.**
- Target and base see the same (ε, λ) pairs, so the paired difference cancels most of the Monte Carlo noise. Diffusion Classifier (Li et al. 2023) uses the same trick.
- Measured on the analytic harness: 6.7× fewer seeds for the same precision.

**λ is stratified, so the error must be too.**
- Each permutation takes one stratum of the band. The standard error uses the collapsed-strata estimator, Var(mean) ≈ Σ_pairs (d_a − d_b)²/K².
- An iid estimate treats the integrand's variation *across* λ as noise and overstated the error roughly 5×.

## 3. Regions and seed contributions

**Noise is spatial, and the mapping back is local enough to use.**
- The Lottery Ticket Hypothesis in Denoising (Mao et al., ECCV 2024) shows that specific blocks of initial noise tend to denoise into specific concepts, and that transplanting a block carries its concept with it.
- Seeds carry composition signatures such as object location and size ("Good Seed").

Three tools, from cheapest to most literal:

1. **Masked band-ELBO.** The squared error sums over latent positions, so weighting it by a face mask splits Δbits exactly into regions. This removes background and style confounds: a LoRA with the same backdrop but a different face scores low in the face region.
2. **Noise → region Jacobian.** ∂(M ⊙ x̂₀)/∂ε, via autodiff, measures how much of the face's prediction comes from noise inside versus outside its footprint. It is needed because attention mixes globally.
3. **Inversion plus region typicality.**
   - Invert the reference and look at z* inside the face footprint.
   - If the weights "know" the face, z* there looks like clean Gaussian noise. If not, the unexplained structure leaks into z*. DIRE and DBINDS exploit this for detection.
   - Note: a *mode* inverts to z* ≈ 0, which is itself atypical, because high-dimensional modes aren't typical points. A real photo is a typical sample of a learned identity, not its mode.

## 4. The Phase 1 trilemma

Pick two of **model-agnostic**, **reference-specific**, **no execution**:

| | What you get |
|---|---|
| Model-agnostic + no execution | Weight-space screening: adapter decoding, spectral entropy and u₁ fingerprints of the conditioning readers, plus metadata/tag matching. It says whether a model is a narrow person/likeness adapter consistent with the reference, not *whose* face it learned. |
| Reference-specific + no execution | Requires per-architecture knowledge of the conditioning pathway (e.g. which linear maps read which text encoder). |
| Model-agnostic + reference-specific | The band-ELBO, region and inversion math written once against `DiffusionBackend` — but it needs an executor per model family. |

Phase 1 ships two things:
- The first row, running on real models.
- The third row's full math, validated on an analytic backend with closed-form ground truth.

Phase 2 adds real executors behind the same protocol.

## 5. Weight-space screening

**Why the conditioning readers.**
- In cross-attention models, the prompt enters through linear K/V projections of the text-encoder output. Fine-tuning only those projections is enough to learn new concepts (Custom Diffusion, Kumari et al. 2023), and closed-form concept editing works on exactly those maps (UCE; TIME).

**Finding them without architecture code.**
- Within a component, the conditioning readers read a width that no module in that component writes, and they come in sibling pairs with identical shapes.
- On a real SDXL LoRA this found all 70 cross-attention layers (140 K/V matrices, width 2048) from shapes alone.
- Joint-attention transformers don't show that structure. They fall back to attention-like projection names, and the report says which method ran.

**Spectral entropy.**
- With pᵢ = σᵢ²/Σσ², H = −Σ pᵢ log pᵢ measures how many directions carry the update.
- Low H means a narrow concept; high H means a broad shift.
- Low-rank updates are analyzed exactly via thin QR, and ΔW is never materialized.

**u₁ fingerprints.**
- The top left singular vector of each update is "what the update injects". Concatenated over the conditioning readers, it screens a LoRA's training subject without inference (0.98–1.0 AUROC across SD1.5, SDXL and FLUX.1 in *Detecting CSAM Text-to-Image LoRAs From Weights*, 2026).
- weights2weights (Dravid et al. 2024) shows identity is linearly encoded in LoRA weight space.
- Scorpion's bank compares u₁ only within a structural family (same module patterns and dims), keyed by `StructuralHash.familyKey`.

## 6. Validation

`scorpion research gmm` and the `GMMTheoryTests` suite run the production probe code against Gaussian-mixture "models". Their noised marginals are closed-form, so the optimal ε-prediction and the exact log-likelihood are known.

A fine-tune that "learned the reference" is the base mixture plus a component for which the reference is a typical draw. Results on the macOS sample image `Owl.heic` (16×16 latent, 32 base components, 256 permutations, λ ∈ [−12, 12]):

| Model | Exact Δbits | Estimated Δbits (90% CI) | Region share | z* atypicality in region |
|---|---|---|---|---|
| learned the reference | +454.7 | +441.4 [+426.6, +456.1] | 100% | 2.0 |
| learned another image | −0.0 | −0.0 | — | 28.7 |
| unchanged | 0.0 | 0.0 (exactly, via CRN) | — | 28.7 |
| base model | | | | 28.7 |

- **Bits by band:** almost all of the learned model's bits sit in λ ∈ [0, 6).
- **Region locality:**
  - In the region-separable model, 100% of ∂x̂₀/∂ε mass lies inside the face footprint.
  - In the coupled model, 46% lies inside a footprint covering 34% of the latent (1.6× concentration).
  - That leakage is the same global-attention effect the Jacobian probe is meant to measure in real models.

The unit tests assert:
- ε̂ equals −σ∇log p_λ (autodiff);
- the estimate is within 10% of the exact ratio;
- identical models cancel exactly;
- a control is measured accurately and ranks below the learned model;
- the face mask captures >95% of a region-only change;
- common random numbers at least halve the variance;
- inversion is more typical when the reference was learned;
- separable regions keep seed contributions local.

## 7. The face-seed needle

**The question.** Segment the face, extract the part of the seed that makes it, and treat the surrounding seed as anything: a needle in a haystack. The formal version is a rare-event probability, so that is how Scorpion treats it.

**Formulation.**
- **Split the seed.** The face's seed footprint F (on the latent grid) splits each seed z into a face segment z_F and a background z_B. With a soft footprint m ∈ [0,1], z = m⊙z_F + √(1−m²)⊙z_B keeps every element N(0,1).
- **Hit.** g(z) is the face fidelity of the latent sampled from z against the reference photo set; a hit is g ≥ t.
- **Needle mass.** μ = P_{z∼N(0,I)}(hit), the probability that a random seed yields the face. Its companions:
  - **Entropy required:** −log₂ μ bits (bits of luck an attacker's seed must supply).
  - **What the weights save an attacker:** Δ = log₂ μ_target − log₂ μ_base.
  - **Attack curve:** 1 − (1−μ)^N, the chance of success within N seeds (the worst case, not just the average).
  - **Background independence ι:** how often a needle face segment survives a fresh surrounding seed. It directly tests "the surrounding seed can be anything".
- **Estimator.** Subset simulation (Au & Beck 2001) with preconditioned Crank–Nicolson moves (Cotter et al. 2013).
  - Chains alternate a pCN move on z_F with an independent fresh z_B, accepted iff still above the level. Both are valid Metropolis steps under N(0,I).
  - Target and base share the seed schedule's streams, so the comparison is paired.
- **Footprint.** It is measured, not assumed:
  - I(p) = E|∂(M⊙x̂₀)/∂ε|ₚ, one denoising step deep or through the whole sampler;
  - F = the face mask plus the highest-influence positions carrying 90% of the influence;
  - |F|/d is the haystack reduction.
- **Face segmentation.** On-device via Vision:
  - the hull of the jaw contour and the brows lifted toward the forehead, intersected with the person matte, feathered;
  - eye, nose and mouth sub-masks per feature ("feature retention" = hits surviving a resampled feature segment).
- **Seeds.** The keyed schedule (HMAC-SHA256 with a per-install key) replaces image-derived seeds.
  - Why: the estimators only need noise independent of the reference, and a fixed schedule gives common random numbers across references, controls and models.
  - A secret key keeps the schedule unpredictable to an adversary. Reports record the key's hash, never the key.
- **Identity test.** Latent fidelity only: no face is ever decoded or rendered.

**Validation against ground truth.** In the toy world the true needle mass is the learned identity's mixture weight π.

| Check | Result |
|---|---|
| Needle mass, identity oracle, π = 10⁻¹…10⁻⁴ | within 0.3 decades of π (4 cases) |
| Fidelity event vs brute-force Monte Carlo (π = 10⁻²) | within 0.3 decades |
| Au & Beck interval vs spread over 10 independent keys | predicted/empirical within 0.4–2.5× |
| Separable face ⊗ background | ι ≥ 0.95; footprint = face block; one-step vs through-sampler IoU ≥ 0.8 |
| Coupled face–background | ι drops; footprint spreads |
| Trigger-gated identity, guidance w = 1, 3.5, 7 | hit rate rises monotonically in w |
| Early exit at λ = 2 | ≥ 95% decision agreement |
| Owl sample, 48 steps | oracle μ̂ 2.05×10⁻³ [6.4×10⁻⁴, 6.5×10⁻³] vs brute force 1.8×10⁻³, π = 10⁻³; base < 1.5×10⁻⁸ |
| Toy experiment (3 identities, 21 pairs) | AUC 1.0 (needle and face Δbits); median ±1.28 bits; ≤ 18.5k denoiser calls per pair; GO |

**What building it taught us** (each is now a test):

1. **The needle mass is a property of (model, sampler).**
   - A few-step sampler over-produces small components: in the toy, 24 steps ≈ 3× π, 48 ≈ 1.8×, 96 ≈ 1.35×.
   - An attacker's sampler and step count are part of the threat model, so reports state the sampler.
2. **Reachability without mass is a knife edge.**
   - Inverting the reference under a model that knows the face round-trips with face MSE 0.006.
   - Under the base it's 0.20 at both 96 and 256 steps. The pre-image exists in exact arithmetic, but numerical error slides it into a neighboring face.
   - Tiny needles (π = 10⁻³) aren't resolved even by first-order inversion under the target. Probability, not reachability, is the measurable quantity.
3. **One photo is photo-level evidence.**
   - When an identity's own photo-to-photo spread (latent MSE ≈ 0.2) matches the gap to the nearest stranger the base can draw (≈ 0.27), the base's near-misses pass a fidelity threshold as often as real identity samples do (30 vs 40 per 4,000). That holds even with a threshold that rejects every one of 108 control photos: model samples aren't distributed like photos.
   - Scoring by **mean distance to several photos** of the person cuts false hits 30× (1 vs 14 per 4,000) and tracks the oracle. A tight identity needs no help.
   - So the reference set is a first-class input. Thresholds sit strictly above every control (Youden's balanced cut is wrong for rare events), and controls must include unrelated faces, not just look-alikes.
4. **Rare events need adaptive MCMC.**
   - With between-level step adaptation only, pCN acceptance collapsed from 0.34 to 0.01 at π = 10⁻⁴ and thresholds stalled.
   - Within-level Robbins–Monro adaptation fixed it (acceptance 0.23–0.38; μ̂ = 6.9×10⁻⁵).
   - Stalls are reported as upper bounds, never as estimates.
5. **Multimodal scores can trap subset simulation.**
   - Closeness to a photo rewards any face that happens to sit near it.
   - On the owl sample the fidelity estimate's interval just misses brute force (4.8×10⁻⁴ [1.3×10⁻⁴, 1.7×10⁻³] vs 1.8×10⁻³).
   - Mitigations: more samples per level, repeated independent runs.
6. **Face-region Δbits is the sturdier discriminator.**
   - In the toy experiment, two of three identity matches had weak needle Δbits under latent fidelity (+1.0, +2.1) but strong face Δbits (+91, +25).
   - The likelihood estimator has no threshold to calibrate. The protocol therefore accepts either statistic for discrimination.

**Limits.**
- Toy results validate estimators and pipeline, not real networks.
- Whether real faces separate under latent fidelity (pose and lighting inflate the photo-to-photo spread) is exactly what the go/no-go experiment decides. Its fallback is a decode-based identity tier.

## 8. Memory pools: finding the reference in what a model memorized

**The question.** One photo is weak evidence of identity (§7, finding 3). But a photo can still be a *seeded endpoint* the model reaches — if the model stored it. Working backwards from Kim & Lee, *Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences* (arXiv 2605.26756): how does unintentional memorization happen, how does it steer later generations, and can we find hints of the reference in the resulting "memory pools"?

**What the paper measures.**
- Memorization is a failure to generalize: the learned local intrinsic dimension falls below the true one (Ross et al., ICLR 2025). *Verbatim* memorization is **coordinate-wise variance collapse**: a subset of coordinates — a template, a logo, a face — stops varying across generations.
- Second-order Tweedie (their Prop. 4.1) turns curvature into variance: Cov[x₀|x_t] = (σ⁴∇²log p + σ²I)/α². In ε-parameterization, with J = ∂ε̂/∂x_λ:

  Var[x₀,ᵢ | x_λ] = (σ²/α²)·ρᵢ,   ρᵢ = 1 − σ·Jᵢᵢ

  ρ is the **retained variance**: 1 = the coordinate is still free at this noise level, 0 = pinned.
- Curvature also fires on naturally low-dimensional data, so they compare against an **underfitted baseline**: the unconditional model (Δh_∅, which must fit everything) or a less-trained checkpoint (Δh_θ̃). A Fisher identity (Prop. 4.2) gives a forward-only surrogate, the squared score difference Δs = (s_c − s_∅)^⊙2 — Wen et al.'s (ICLR 2024) magnitude detector, per coordinate.
- On SD 1.4/2.1 (DDIM-50, w = 7.5, final step, Hutchinson K = 16): localization IoU ≈ 0.90–0.95, detection AUC ≈ 0.997. Their stated limit: concept-level memorization that spreads over the image doesn't localize.

**How it happens (the mechanism, from the literature the paper builds on).**
- The optimal denoiser for a finite training set sends every trajectory into one training point: the "collapse" regime, where the backward trajectory falls into a training point's attractor (Biroli et al., *Dynamical regimes of diffusion models*, Nat. Commun. 2024).
- Networks reach it only when trained past τ_mem, which grows linearly with the training-set size n; before that there is a generalization window (Bonnaire et al., NeurIPS 2025).
- So memorization is *unintentional* whenever the effective n per condition is small: duplicated photos (Carlini et al. 2023; Somepalli et al. 2023), few-image fine-tunes run for thousands of steps (exactly an identity LoRA), and unique caption or trigger tokens — conditional overfitting, where p(x|c) collapses before p(x) does (Zhai et al., CLiD, NeurIPS 2024; also the paper's baseline argument).
- Across a person's photos the face is the invariant and the background varies, so an identity fine-tune memorizes *locally*, on face coordinates: the paper's template-verbatim case with the face as the template.

**How it steers later generations.**
- **Attractor:** once x_t is in a pool's basin, the rest of the noise no longer matters for the pinned coordinates — many seeds give the same face (large needle mass μ) while the background stays free (ι ≈ 1).
- **Recurrence:** generations come back as near-copies of the few stored photos (the Carlini-style extraction criterion).
- **Leakage:** replication triggers are spread across text-embedding space (Kowalczuk et al., *Finding Dori*, 2025) and guidance amplifies the conditional pull, so prompts that never ask for the person can land in the pool.
- **Seeds matter:** seeds near sharp regions are drawn into memorized basins (Jeon et al., ICML 2025, SAIL).

**Two tiers.** Memorization stores points, not concepts, so one photo is exactly the right evidence for it:
- **Tier M (memory pools, one photo suffices):** this photo — or other photos of this face — sits in a collapsed pool; here is its seed endpoint and how robustly seeds reach it.
- **Tier L (likeness: needle and face Δbits, §7):** this face is generable. A well-generalized identity leaves no localized collapse (the paper's own limit), so Tier M never replaces Tier L.

**Formulation** (`MemoryPoolProbe`, all through `DiffusionBackend`, decode-free). The probe evaluates at the *noised reference* x_λ = αx_ref + σε (keyed ε, a grid of λ) instead of the paper's generated samples — Scorpion never generates for display, and the reference is what's on trial.
1. **Is there a pool here?** Collapse C = ρ_base − ρ_target per coordinate and region (the paper's Δh_θ̃ × σ²), and C_∅ = ρ_uncond − ρ_cond under the target (Δh_∅ × σ²; no base needed). diag J by Hutchinson (one VJP per Rademacher probe, probes shared across models so differences are paired). The pool signal is the collapse **sustained over two adjacent noise levels at which the base has released the face** (0.5 ≤ ρ_base ≤ 1.1) — see findings 2 and 4.
2. **Is it yours?** Where the target denoises the noised photo: back onto itself (**snap** = base distance / target distance; its geometric mean over released levels is the single-photo membership statistic), onto the person (calibrated latent fidelity), or onto someone else.
3. **Can a seed reach it?** Invert the reference to z*, move the face segment by β (pCN), resample the surroundings, count returns: r(β). **r(0) is reachability, r(1) the needle mass**; β½ is the pool's radius in seed space.
4. **Does it take over other generations?** Needle mass under trigger, descriptor and empty prompts, and the share of the person's hits that recur as near-copies from clearly different seeds.

Readout (rule-based, uncalibrated): `memorized-photo` (pool + snap ≥ 4 + wider basin), `identity-pool` (pool pulling onto the person, or recurring copies), `nearby-pool` (a stored face near yours, not yours), `generalized-likeness` (no pool, but the needle sees the face), `no-evidence`.

**A toy with known memorization** (`MemoryScenario`). The target holds an identity as a matched-variance kernel density over n training photos: components at μ̂ + c(xᵢ − μ̂) with variance h² = (1 − κ)s², c = √(1 − h²/s²). **κ = 0** is one generalized Gaussian (learned); **κ = 1** is the photos themselves (stored). Total variance stays s² at every κ, so only *how* the identity is held changes. Dials for the mechanisms: `duplicates`, `scope` (face vs whole photo), and a trigger whose conditional model is memorized while the unconditional one keeps a smoothed identity at weight `leak`. The exact diagonal Hessian and posterior variance of a Gaussian mixture are closed-form (`exactDiagHessian`, `exactPosterior`).

**Validation against ground truth.**

| Check | Result |
|---|---|
| Closed-form GMM curvature vs autodiff (λ = 0, 4, 8) | relative error < 10⁻³ |
| Prop. 4.1 (Tweedie) vs independently computed mixture posterior | \|Δρ\| < 10⁻³ |
| Hutchinson region means (K = 64) | within 0.02 of exact; owl κ ≥ 0.5 at the peak: 0% error (K = 16) |
| Memorized face localizes | IoU ≥ 0.9 (exact and K = 16), ≥ 0.8 forward-only; background Δρ < 0.05; a whole-photo memory delocalizes |
| Collapse band | peaks between the base's natural variability and the pool width (−log v … −log h²) |
| Dose-response | face collapse 0 < 0.17 < 0.50 < 0.81 for κ = 0, 0.5, 0.9, 1 (owl, sustained: 0.00, 0.15, 0.46, 0.68) |
| Duplication (×5) | photo-level mass ×2–5 (theory 3.3); collapse no weaker |
| Conditional overfitting | C_∅ ≥ 0.3 under a trigger; exactly 0 when every prompt sees the memory |
| Single-photo membership (band snap) | AUC 1.0 at κ = 1, chance at κ = 0; owl 0.97–1.00 for κ ≥ 0.5 |
| Held-out photo of a memorized person | `identity-pool` via recurring copies (owl: 25% at κ = 0.9, 56% at κ = 1); ≤ 5% copies when learned |
| Basin | r(1) matches brute-force μ within its CI; owl κ = 1: β½ 0.50 vs base 0.00 |
| Leakage | empty-prompt capture rises with leak; stored photos recur (≥ 50% copies) |
| Paper's limit | learned identity (κ = 0): no sustained collapse, yet face Δbits > 10 when the face is new to the base |
| Tier M dry run (3 identities × 4 κ, 96 references, 22 s) | GO: membership AUC 1.0, IoU 1.0, Spearman 1.0, 0% false pools at κ = 0 |

**What building it taught us** (each is now a test):

1. **One photo is enough for memorization — and only for memorization.**
   - Member vs held-out AUC is 1.0 when photos are stored and chance when the face is only learned: with nothing photo-specific stored there is nothing photo-specific to find.
   - A held-out photo of a memorized person *cannot* be reproduced exactly (basin r ≡ 0): a memorizing model's density is spiky, low between stored photos — the generalization gap membership inference feeds on. Its evidence is recurrence: the person's generations come back as copies of other photos.
2. **The baseline decides what "collapse" means.**
   - Against the pre-fine-tune base, a *learned* face the base has never seen reads as collapse (owl, κ = 0: 0.26): the base hesitates between faces at high noise, which is extra variance on its side, not memorization on the target's.
   - Fix: count only noise levels where the base has released the face within a single mode (0.5 ≤ ρ_base ≤ 1.1), and require the collapse to persist over two adjacent levels. Learned-face false pools went from 50% to 0%.
   - The paper's remedy is cleaner: a baseline that already knows the face. The unconditional branch read exactly 0.00 where the base comparison read 0.32. In Phase 2, prefer a less-trained checkpoint or the unconditional branch (C_∅) when available.
3. **The forward-only surrogate conflates learning with memorization at the reference.** (ε̂_T − ε̂_B)² was 0.15 for a learned face vs 0.64 stored — both "detected". A forward-only *variance* estimate (the spread of x̂₀ across noise draws, ×α/σ) tracks the exact collapse for stored photos (0.79 vs 0.81) but is confounded for faces new to the base, where x̂₀ switches between base faces. Prefer Hutchinson VJPs; without autodiff, trust only memorized-photo readouts backed by snap and basin.
4. **Memorization is band-limited in noise.** A pool pins the face from the base's natural variability down to its own width (λ ≈ 3…8 here) and releases above. The paper's final-step choice works when pools are points; a finite-width pool is invisible at t → 0. A base's hesitation, by contrast, occupies one narrow band — hence the two-level rule.
5. **Held-out photos meet pools only in a narrow window.** At moderate noise a held-out photo falls into its person's pool (collapse at λ ≈ 2–5); at low noise the broad generic components win. Collapse at the photo is fragile evidence for identity pools; recurrence is robust.
6. **Likelihood gain measures what the fine-tune added, not what is generable.** The same learned identity (κ = 0) gained 2 bits in a toy world where a crowd face already sat next to the reference, 46 in another, and 862 for the owl, a face unlike anything its base knows. Storing the photo added more on top (owl: +1,166 at κ = 1). Whether the face can be generated is the needle's question (Tier L).

**Limits.**
- Toy memories are Gaussian pools; real ones have structure (partial and template memorization, attention-mediated leakage).
- Triggers outside the prompts we try can open a pool (Finding Dori); absence under our prompts is not absence.
- No pool is not the same as not generable: a well-learned identity is Tier L's case.
- Uncalibrated tiers until the Phase 2 memory arm (Docs/EXPERIMENT.md) runs on real checkpoints.

**Boundaries.** Reference-directed only: the probe asks whether *this* reference sits near a pool, and Scorpion does not enumerate pools without a reference (that is training-data extraction). Decode-free: statistics and latent-grid maps only.

## 9. Code map

| Concept | File |
|---|---|
| Partial download (header + ranges) | `Fetch/RangeFetcher.swift`, `Formats/Safetensors.swift`, `Weights/TensorInventory.swift` |
| Adapter decoding | `Weights/AdapterDecoder.swift` |
| Spectral entropy, u₁ | `Weights/SpectralStats.swift` |
| Conditioning readers, family key | `Weights/StructuralHash.swift` |
| Seed permutations, log-SNR bands | `Seeds/SeedEngine.swift` |
| Δbits, regions, CRN, stratified SE | `Probes/BandLikelihoodProbe.swift` |
| Seed inversion, typicality | `Probes/InversionProbe.swift` |
| Noise→region Jacobian | `Probes/RegionSensitivityProbe.swift` |
| Ground-truth backend | `Probes/AnalyticGMMBackend.swift`, `Probes/GMMResearch.swift` |
| Executor protocol (Phase 2 plugs in here) | `Probes/DiffusionBackend.swift` |
| Keyed seed schedule, prompt sets | `Seeds/SeedSchedule.swift`, `Seeds/SeedEngine.swift` |
| Face and feature segmentation | `Reference/FaceSegmentation.swift` |
| Deterministic sampler (DDIM, DPM-Solver++ 2M, CFG, early exit) | `Probes/DeterministicSampler.swift` |
| Seed footprint | `Probes/SeedFootprint.swift` |
| Rare-event estimation | `Probes/SubsetSimulation.swift` |
| Face-seed needle | `Probes/NeedleProbe.swift`, `Probes/NeedleScenario.swift`, `Probes/NeedleResearch.swift` |
| Experiment protocol runner | `Probes/Experiment.swift`, `Docs/EXPERIMENT.md` |
| Curvature: Hutchinson diag J, retained variance, exact GMM Hessian and posterior | `Probes/Curvature.swift` |
| Memory pools (collapse, snap, basin, capture, tiers) | `Probes/MemoryPoolProbe.swift`, `Probes/MemoryScenario.swift`, `Probes/MemoryResearch.swift` |

## 10. Relation to Obscur (Frigate)

Obscur attaches per-corpus KV banks to FLUX.2 Klein through a decoupled cross-attention branch and records, per entry, layer and denoising step, how much attention mass a generation spent on each registered image. That is royalty attribution: *what the model read*.

Scorpion asks the reverse question of a third-party model: *what could these weights reproduce?* It reuses the same patterns (content-addressed entries, seeded determinism, dimensionless normalization, versioned reports) and the same MLX stack.
- [Frigate @ a19b127](https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2) — ObscurKit: KV banks, composition, attribution recorder
- [Frigate @ 1d3455e](https://github.com/rao-studios/Frigate/commit/1d3455e9b55e9ce01b093f7546c802cc88e7a040) — RMS-matched adapter branch, influence sweep

In Phase 2, a FluxKit executor can report both at once: Δbits for the reference, and Obscur-style attention mass on reference-derived tokens at the hooked sites.

## 11. Limitations and open questions

- **Identity specificity.**
  - Without an executor, signals are attribute-level. CLIP descriptions capture hair, eyewear, framing and medium, not identity.
  - Whether spectral narrowness separates identity adapters from style adapters in the wild is untested until a labeled corpus exists. The component carries low provisional weight.
- **Calibration.** "X% likelihood" needs labeled (model, reference, match) pairs; `scorpion eval` fits it. Until then, reports say `calibrated: false`.
- **ELBO versus exact likelihood with real networks.**
  - The identity in §2 is exact for optimal denoisers.
  - Trained networks are not optimal, so Δbits becomes a difference of variational bounds. It is still the standard membership-inference and diffusion-classifier statistic.
- **Joint-attention transformers.** The read-only-width structure is absent there, so conditioning-reader selection falls back to names. Spectral signals are weaker than for cross-attention UNets.
- **Formats.** Pickle checkpoints are refused by design. GGUF is inventoried but its quantized payloads are not decoded. ONNX is not analyzed.

## References

- Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021 — https://arxiv.org/abs/2011.13456
- Kingma et al., *Variational Diffusion Models*, 2021 — https://arxiv.org/abs/2107.00630
- Kong, Brekelmans, Ver Steeg, *Information-Theoretic Diffusion*, ICLR 2023
- Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019 — https://arxiv.org/abs/1810.09136
- Li et al., *Your Diffusion Model is Secretly a Zero-Shot Classifier*, ICCV 2023 — https://arxiv.org/abs/2303.16203
- Wen et al., *Tree-Ring Watermarks*, NeurIPS 2023 — https://arxiv.org/abs/2305.20030
- Yang et al., *Gaussian Shading*, CVPR 2024
- Mokady et al., *Null-text Inversion*, CVPR 2023
- Wang et al., *DIRE for Diffusion-Generated Image Detection*, ICCV 2023 — https://arxiv.org/abs/2303.09295
- *DBINDS: Can Initial Noise from Diffusion Model Inversion Help Reveal AI-Generated Videos?*, 2025 — https://arxiv.org/html/2511.09184v1
- Mao, Wang, Aizawa, *The Lottery Ticket Hypothesis in Denoising*, ECCV 2024 — https://arxiv.org/abs/2312.08872
- *Good Seed Makes a Good Crop: Discovering Secret Seeds in Text-to-Image Diffusion Models*, 2024 — https://arxiv.org/abs/2405.14828
- Choi et al., *Perception Prioritized Training of Diffusion Models*, CVPR 2022
- Kumari et al., *Multi-Concept Customization of Text-to-Image Diffusion* (Custom Diffusion), CVPR 2023
- Gandikota et al., *Unified Concept Editing in Diffusion Models*, WACV 2024
- Dravid et al., *Interpreting the Weight Space of Customized Diffusion Models* (weights2weights), 2024 — https://arxiv.org/abs/2406.09413
- *Detecting CSAM Text-to-Image LoRAs From Weights*, 2026 — https://arxiv.org/html/2607.25750v1
- Hugging Face, safetensors metadata parsing over HTTP Range — https://huggingface.co/docs/safetensors/metadata_parsing
- Au & Beck, *Estimation of small failure probabilities in high dimensions by subset simulation*, Probabilistic Engineering Mechanics 2001 — overview: https://arxiv.org/pdf/1505.03506
- Cotter, Roberts, Stuart, White, *MCMC methods for functions: modifying old algorithms to make them faster* (pCN), Statistical Science 2013
- Xu et al., *Subset simulation coupled with pCN-MCMC*, Water Resources Research 2024 — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR038260
- Lu et al., *DPM-Solver++*, 2022 — https://arxiv.org/abs/2211.01095
- Kim & Lee, *Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences*, 2026 — https://arxiv.org/html/2605.26756v1
- Ross et al., *A Geometric Framework for Understanding Memorization in Generative Models*, ICLR 2025
- Kamkari et al., *A Geometric View of Data Complexity: Efficient Local Intrinsic Dimension Estimation with Diffusion Models* (FLIPD), NeurIPS 2024
- Wen et al., *Detecting, Explaining, and Mitigating Memorization in Diffusion Models*, ICLR 2024 — https://openreview.net/forum?id=84n3UwkH7b
- Biroli, Bonnaire, de Bortoli, Mézard, *Dynamical Regimes of Diffusion Models*, Nature Communications 2024 — https://arxiv.org/abs/2402.18491
- Bonnaire, Urfin, Biroli, Mézard, *Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training*, NeurIPS 2025 — https://arxiv.org/abs/2505.17638
- Jeon et al., *Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes*, ICML 2025 — https://arxiv.org/abs/2412.04140
- Zhai et al., *Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy*, NeurIPS 2024 — https://arxiv.org/abs/2405.14800
- Kowalczuk et al., *Finding Dori: Memorization in Text-to-Image Diffusion Models Is Less Local Than Assumed*, 2025 — https://arxiv.org/abs/2507.16880
- Asthana & Belagiannis, *Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability*, ICLR 2026 — https://arxiv.org/abs/2601.20642
- Carlini et al., *Extracting Training Data from Diffusion Models*, USENIX Security 2023
- Somepalli et al., *Diffusion Art or Digital Forgery?* (CVPR 2023) and *Understanding and Mitigating Copying in Diffusion Models* (NeurIPS 2023)
- Webster, *A Reproducible Extraction of Training Images from Diffusion Models*, 2023
