# The face-seed needle experiment (go/no-go)

Everything the needle probe claims about real models hangs on one experiment (the memory-pool tier has its own arm, below):

> Does needle mass (or face-region Δbits) separate a LoRA's identity from look-alikes and strangers, with a local seed segment, at an affordable cost?

This document is the protocol. `scorpion experiment run --manifest …` implements its analysis; today it runs a synthetic dry run (`Docs/examples/toy-experiment.json`). It runs on real models once a Phase 2 executor is registered with `BackendRegistry`.

## Setup

**Models**
- **Base:** FLUX.2 Klein through a Frigate FluxKit executor (first family), then others.
- **Identity LoRAs:** three identities that are **consenting or synthetic**. Rank-16 LoRAs, each trained on 20 photos, with 5 held out per identity.
- **Model negatives:** one style LoRA and one object LoRA.

**References**

| Set | Use | Label |
|---|---|---|
| Held-out photos of each identity (reference sets of 3–5 photos) | positives | `identity-match` |
| ≥ 10 attribute-matched consenting look-alikes per identity | hard negatives | `look-alike` |
| ≥ 20 unrelated faces | negatives | `unrelated` |

**Controls** for threshold calibration are look-alikes *and* unrelated faces. The toy work showed that thresholds calibrated on look-alikes alone let the crowd through.

## Pairs

- Every identity LoRA × (its held-out photos, its look-alikes, unrelated faces).
- Every style/object LoRA × identity photos.

## Measurements per pair

| Measurement | Probe |
|---|---|
| Needle mass μ, Δbits (target vs base, paired), 90% CI | `NeedleProbe` (subset simulation, latent fidelity with a reference set, mean distance) |
| Attack curve 1 − (1−μ)^N at w ∈ {1, 3.5} | `NeedleProbe` |
| Background independence ι, feature retention (eyes/nose/mouth) | `NeedleProbe` |
| Seed footprint \|F\|/d and captured influence | `SeedFootprintProbe` |
| Face-region Δbits and sub-bands | `BandLikelihoodProbe` (face mask) |
| IoU of the per-position Δbits map with feature masks | `BandLikelihoodProbe.perPositionMap` |
| Early-exit agreement at λ = 2 | `NeedleProbe` |
| Cost: denoiser calls, wall clock | runner |

The sampler and step count are recorded with every result. The needle mass is defined for a sampler; in the toy, few steps over-produce small components (24 steps ≈ 3× π).

## Go/no-go

1. **Discrimination:** AUC(identity-match vs look-alike) ≥ 0.90 on needle Δbits **or** face Δbits.
2. **Locality:** median \|F\|/d ≤ 0.35 at 90% influence **and** median ι ≥ 0.70. Otherwise "the surrounding seed can be anything" fails for this family.
3. **Precision and cost:** median bits CI half-width ≤ 1.5 within ≤ 20,000 denoiser calls per pair.
4. **Early exit:** decision agreement ≥ 0.95.

GO requires all four.

## Fallbacks

| Failure | Fallback |
|---|---|
| Low ι | Joint needle over the full seed (F = everything) |
| Latent fidelity too photo-bound (look-alike AUC low, face Δbits fine) | Report face Δbits; revisit a decode-based identity tier |
| Subset simulation traps (intervals miss spot-check brute force) | More samples per level; repeated independent runs |
| Too costly | 256² face crops, fewer steps (after checking sampler bias), smaller N |

## Memory arm (Tier M: is the photo, or the face, stored?)

A separate verdict from the needle's GO above. Neither replaces the other: a well-learned identity is generable without a memory pool, and a stored photo is a harm of its own (the model can regurgitate it). Theory and toy validation: RESEARCH.md §8.

**Models**
- The same identity LoRAs, **checkpointed at 250, 1k, 4k and 10k steps**. Memorization sets in with training time (τ_mem ∝ n), so checkpoints are the real-model version of the toy's κ dial. Early checkpoints double as the paper's *less-trained baseline*.
- One run with **duplicated photos** (the first training photo ×5).
- One **full fine-tuned checkpoint with no known base**, to test the unconditional-branch comparison (C_∅) on its own.

**References**

| Set | Label |
|---|---|
| Training photos of each identity | `member` |
| Held-out photos of the same identity | `held-out` |
| Look-alikes, unrelated faces | `look-alike`, `unrelated` (specificity) |

**Measurements per reference** (`MemoryPoolProbe`): sustained face collapse against each available baseline (base, early checkpoint, unconditional branch), band snap, IoU of the collapse map with the face and feature masks, seed-endpoint basin r(β) and β½, capture under trigger, descriptor and empty prompts with the recurring-copy rate, tier, cost (VJPs and denoiser calls).

**Go/no-go (Tier M)**
1. **Membership:** AUC(member vs held-out) ≥ 0.90 on band snap at the latest checkpoint.
2. **Localization:** median IoU(collapse map, face) ≥ 0.50 for members in a pool.
3. **Dose-response:** Spearman(training steps, median member collapse) ≥ 0.80.
4. **Specificity:** ≤ 10% of photos flagged as pools at the earliest checkpoint, and no look-alike or unrelated face flagged at any checkpoint.

**Fallbacks**

| Failure | Fallback |
|---|---|
| Base comparison flags learned faces (specificity) | Use the early checkpoint or the unconditional branch as the baseline (the paper's remedy) |
| Collapse doesn't localize (template or partial memories) | Report per-feature shares and the basin; revisit the region definition |
| VJPs too costly | Hutchinson K = 1–4 (the paper found K = 1 competitive), fewer noise levels, λ ≥ 2 only |

Toy dry run (`Docs/examples/toy-memory.json`: 3 identities × κ ∈ {0, 0.5, 0.9, 1}, 96 references, about 22 s):

```
Tier M · memory pools · GO
  κ      label      n   median collapse   median band snap   in a pool   median IoU
  0.00   member     12       0.00                 1.8             0%        0.38
  0.50   member     12       0.15                 2.7            92%        1.00
  1.00   member     12       0.68                57.3           100%        1.00
  1.00   held-out   12       0.32                 0.9            75%        0.89
PASS  membership: 1.000
PASS  localization: 1.00 (12 pools)
PASS  dose-response: 1.00
PASS  specificity: 0%
```

## Ethics

- Consenting or synthetic identities only.
- Scorpion is decode-free: latents never become images. Outputs are statistics only.
- Seed schedules are keyed; reports carry the key's hash, not the key.
- The memory arm is reference-directed: it asks whether given photos sit in a pool and never enumerates what a model stored.

## Running

```sh
swift build && scripts/build-metallib.sh
.build/debug/scorpion experiment run --manifest Docs/examples/toy-experiment.json [--json report.json]
.build/debug/scorpion experiment run --manifest Docs/examples/toy-memory.json
```

Manifest fields:
- `toy`: synthetic world — identities, photos per identity, look-alikes and their distance, unrelated people, needle weight π, identity spread, grid size, style model; `needleArm` (default true) and `memorization {kappas, members, duplicates, identityWeight, references, probes}` for the Tier M arm.
- `settings`: samples per level, max levels, sampler steps, guidance, face band, face-probe permutations.
- `pairs`: real pairs `{id, backend, base, references, controls, label}` for Phase 2 executors.

Toy dry run (3 identities, 21 pairs; about 22 s in debug) — AUC 1.0 on both statistics, median ±1.28 bits, ≤ 18.5k calls per pair, GO:

```
PASS  discrimination: 1.000
PASS  locality: |F|/d 0.25, ι 1.00
PASS  precision & cost: ±1.28 bits, max 18528 calls
PASS  early exit: 0.960
```

The dry run validates the pipeline and the verdict logic, not real networks. In it, two of three identity matches had weak needle Δbits under latent fidelity (+1.0, +2.1) but strong face Δbits (+91, +25). That is the pattern to watch for on real models, and why criterion 1 accepts either statistic.
