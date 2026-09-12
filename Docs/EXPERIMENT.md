# The memorization experiment (go/no-go)

Everything Scorpion's heatmap claims about real models rests on one experiment:

> Does the memorization probe separate a model's training images from held-out images of the same subject, localize what it stored, grow with training, and stay quiet when nothing is stored — at an affordable cost?

This document is the protocol. `scorpion experiment run --manifest …` implements its analysis. Today it runs a synthetic dry run (`Docs/examples/toy-memory.json`); the real arm uses `scorpion memorization --backend flux2-klein` per reference until the runner drives executors directly.

## Setup

**Models**
- **Base:** FLUX.2-klein-base-4B, through the `ScorpionFlux2` executor. The distilled Klein never visits the noise levels the probe needs; see RESEARCH.md §4.
- **Identity and object LoRAs:** three identities and two objects, all **consenting or synthetic**. Each is a rank-16 LoRA trained on 20 images, with 5 held out.
- **Checkpoints:** each LoRA is saved at **250, 1k, 4k and 10k steps**. Memorization sets in with training time (τ_mem ∝ n), so checkpoints are the real-model version of the toy's κ dial. Early checkpoints double as the paper's *less-trained baseline*.
- **Duplication:** one run duplicates a single training image ×5.
- **Negative models:** one style LoRA, which has no stored images of these subjects.

**References**

| Set | Label |
|---|---|
| Training images of each LoRA | `member` |
| Held-out images of the same subject | `held-out` |
| Look-alikes, unrelated images | specificity |
| ≥ 19 held-out / unrelated images per LoRA | **negative controls** (calibrate p-values) |

## Measurements per reference

Every measurement comes from the `evidence` profile of `MemorizationProbe`:
- region masses and p-values against the controls;
- the membership snap, measured on confirmation draws;
- the region's peak λ and mean collapse;
- IoU of the counted regions with the image's stored content (the whole image for members; the subject mask where one is annotated);
- the verdict;
- cost: evaluations and wall clock.

The optional seed basin (`--basin`) runs on members at the latest checkpoint.

## Go/no-go

1. **Membership:** AUC(member vs held-out) ≥ 0.90 on membership snap at the latest checkpoint.
2. **Localization:** median IoU(counted regions, stored content) ≥ 0.50 for members with a region.
3. **Dose-response:** Spearman(training steps, median member collapse) ≥ 0.80.
4. **Specificity:** at the earliest checkpoint, ≤ 10% of images are flagged; no look-alike or unrelated image is flagged at any checkpoint; no significant region appears for the style LoRA.

GO requires all four.

## Fallbacks

| Failure | Fallback |
|---|---|
| Base comparison flags learned content (specificity) | Use the early checkpoint as the baseline, or the unconditional branch (prompt collapse), as the paper recommends |
| Collapse doesn't localize (template or partial memories) | Report per-region shares and the basin; revisit the region rule |
| Too costly | Hutchinson K = 1–2 (the paper found K = 1 competitive), fewer levels, the `quick` profile for screening |
| Controls look like members | Use more distant controls; report the control distribution with every result |

## Toy dry run

`Docs/examples/toy-memory.json` runs 3 subjects × κ ∈ {0, 0.5, 0.9, 1}, 96 references, in about 25 s:

```
Scorpion experiment · toy memorization: 3 synthetic subjects at four memorization strengths · GO
  κ      label      n   median collapse   median snap   flagged   median IoU
  0.00   member    12              0.00           1.0        0%         0.00
  0.50   member    12              0.16           3.1      100%         0.74
  0.90   member    12              0.46          11.4      100%         1.00
  1.00   member    12              0.68          63.6      100%         1.00
  1.00   held-out  12              0.46           1.0       67%         0.31
PASS  membership: 1.000
PASS  localization: 1.00 (36 regions)
PASS  dose-response: 1.00
PASS  specificity: 0%
```

Held-out images of a memorized subject are flagged `memorized-other-content`. The model pins them onto its stored images of that subject, which is the correct reading. Their snap stays near 1, so they are never called `memorized`.

The dry run validates the pipeline and the verdict logic, not real networks.

## Ethics

- Consenting or synthetic subjects only.
- Decode-free: latents never become images. The heatmap is drawn over the reference the user supplied.
- Reference-directed: the probe asks whether given images are stored, and never enumerates what a model stored.
- Seed schedules are keyed. Reports carry the key's hash, not the key.
