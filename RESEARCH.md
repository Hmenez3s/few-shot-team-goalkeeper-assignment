# Reference-conditioned football classification: robustness study

This study classifies pre-detected crops as player A/B or goalkeeper A/B from labeled examples earlier in the same half. It evaluates a specific four-class protocol. It does not introduce a new architecture, learning rule, detector or tracker.

## Reference sensitivity and appearance baselines

On the official test (2,671 queries across three matches), each method received the same randomly selected crops per draw. The one-reference condition sampled one of the six per class. The extra crops show the same outfield reference athlete, not six different athletes. Outfield queries exclude that athlete, and all references precede the queries. No draw or checkpoint was selected by test performance.

| Encoder or comparator | 1 ref: accuracy ± SD | 6 refs: accuracy ± SD | 1 ref: macro recall ± SD | 6 refs: macro recall ± SD |
|---|---:|---:|---:|---:|
| ResNet-18 tuned with one reference | 90.49 ± 2.18 | **92.29 ± 0.24** | 88.13 ± 2.14 | 91.37 ± 0.53 |
| ResNet-18 tuned with six references | 89.92 ± 1.29 | 91.38 ± 0.23 | 87.11 ± 2.64 | **91.83 ± 0.98** |
| Frozen ImageNet ResNet-18 | 52.05 ± 4.84 | 69.78 ± 2.08 | 48.98 ± 4.01 | 67.63 ± 2.34 |
| Torso HSV histogram, nearest prototype | 73.44 ± 6.81 | 82.01 ± 3.79 | 71.90 ± 5.65 | 83.27 ± 2.92 |
| Torso HSV, reference-seeded clustering | 68.11 ± 8.08 | 72.86 ± 5.43 | 65.83 ± 6.52 | 71.82 ± 5.72 |
| Frozen OSNet x1.0, Market-1501 | 57.07 ± 4.51 | 75.55 ± 2.53 | 59.57 ± 4.93 | 78.78 ± 2.59 |

Values are percent with sample SD across 30 draws on the same three matches, not intervals for generalization. Clustering is transductive: it receives all unlabeled query crops of a session, unlike the other comparators. OSNet is a generic person ReID encoder. A sports-specific PRTReID comparison is not reported: its published recipe uses SoccerNet and the match provenance of its applicable weights was not established here.

For the same one-reference-trained weights, increasing support from one to six improved mean accuracy by 1.80 percentage points and reduced draw SD from 2.18 to 0.24 points. The range narrowed from 83.79–92.55% to 91.69–92.62%. Training with six did not yield the highest overall accuracy, but did yield the best macro recall with six references. These are descriptive results on a previously used test split, not a fresh confirmatory evaluation. The original fixed one- and six-reference confusion matrices were reproduced exactly. See [per-draw metrics, reference indices and input hashes](results/research-reference-draws.json).

## Training seeds and match variation

The one-reference model ran on seeds 13, 25, 29, 42 and 57, with an independent checkpoint choice on validation per run and fixed test references. The official test gave **90.72% ± 1.11 points** (88.96–91.65%) top-1 and **87.53% ± 1.88 points** macro recall. Each run used eight epochs and 180 episodes per epoch.

| Held-out games | Query boxes | Top-1 | Macro recall |
|---|---:|---:|---:|
| Official test: 7, 8, 11 | 2,671 | 91.65% | 88.16% |
| Rotated fold: 2, 3, 5 | 3,193 | 87.54% | 89.09% |
| Rotated fold: 4, 6, 9 | 3,290 | 85.53% | 84.50% |
| **Pooled, nine games** | **9,154** | **88.02%** | **87.10%** |

The folds rotate the three official game groups between training, validation and test, using seed 25. Each match is evaluated only by a model fitted and selected on other matches. Equal-game mean accuracy is 88.08%. This reuses the same source dataset and is not external validation. See [seed and fold results](results/research-seeds-and-folds.json).

## External holdout and contribution

- SoccerNet GSR 2024 and 2025 have identical SHA-256 identifiers for all four archives. The `challenge.zip` archive index contains no `Labels-GameState.json` annotations. Neither supplies more labeled games for this evaluation.
- [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2) is an external candidate. The actual file `gsr/118577/118577_1st.json` returns `401 GatedRepo` without an approved account. After obtaining access, check the shipped annotation schema, recording independence and coverage of all four classes before scoring.
- [Prototypical Networks](https://arxiv.org/abs/1703.05175), [game-agnostic team discrimination](https://arxiv.org/abs/1907.01058) and [PRTReID](https://arxiv.org/abs/2401.09942) bound the novelty claim. The result is an empirical study of reference sensitivity and the tradeoff with simpler baselines. [The INT8 measurements](results/formats.json) describe deployment, not a new quantization method or a matched latency comparison against the appearance baselines.

## Reproduce

Use the prepared data, build the reference bank with `reference_bank.py`, and run:

```bash
python research_references.py --root DATA --bank DATA/work/references \
  --one models/one-reference.pth --six models/six-references.pth \
  --out NEW_JSON --device mps --draws 30 --reid-torchscript LOCAL_OSNET_SCRIPT
```

Use `osnet_x1_0` with the Market-1501 weights listed in the official Torchreid model zoo (Google Drive file ID `1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA`). Instantiate with `num_classes=751, pretrained=False`, load the state dict strictly, set eval mode and trace its `[N, 512]` embeddings with an RGB `[1, 3, 256, 128]` input. The comparison receives ImageNet-normalized inputs at this resolution. Run `train.py --root ISOLATED_DATA --seed SEED` for each seed. For the rotated folds, remap the split of each whole official game group in an isolated input manifest before training. The original archives and models were not modified.
