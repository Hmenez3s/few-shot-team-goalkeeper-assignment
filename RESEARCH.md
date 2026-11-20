# Reference-conditioned football classification: evaluation

This is an empirical study of classifying supplied football boxes as outfield player A/B or goalkeeper A/B using labeled crops from the same half. It evaluates reference selection, generalization across held-out matches, and deployment. The classifier does not do detection or tracking.

## Ground truth and evaluation

SoccerNet GSR defines `left` and `right` relative to the camera, not as team IDs across clips. The preprocessing now aligns outfield slots within each half using the agreement of jersey numbers. Among 164 clips in 18 halves, five outfield assignments were swapped: `SNGS-092`, `111`, `126`, `131`, and `197`. A visual audit of the goalkeeper kits in those clips and neighboring clips supported retaining their original slots; the outfield swap is not applied to goalkeepers. Source archives were not changed. The preparation audit records each assignment and its support. No identical crop was found across the three game-disjoint splits.

The **retrospective** protocol chooses references and filters queries using later jersey labels to evaluate outfield athletes other than the reference athlete. Its figures are not deployment estimates. The **prospective replay** fixes a timestamp-only bootstrap and query-frame schedule; it uses only early annotations to pick references and does not reject later labeled player or goalkeeper boxes based on jersey. The two protocols have different query populations. Do not compare scores across these two protocols as if they were computed on identical cases.

## Retrospective results

All fixed test metrics below use the same 2,659 queries in games 7, 8, and 11. Checkpoints were selected by validation macro recall.

| Fixed-reference condition | Accuracy | Macro recall |
|---|---:|---:|
| Frozen ImageNet ResNet-18, one reference per class | 55.28% | 48.06% |
| Tuned ResNet-18, one reference per class | 95.60% | 86.10% |
| Tuned ResNet-18, six references per class | 96.77% | 93.81% |

Five one-reference training seeds on that same split averaged **95.92% ± 0.70 pp** accuracy and **89.00% ± 2.86 pp** macro recall (sample SD). Three rotated folds holding out whole games gave **89.11% accuracy and 85.04% macro recall over 9,130 boxes in nine games**. Fold accuracies were 95.60%, 87.15%, and 85.77%. These are internal SoccerNet folds, not an external-dataset evaluation. SoccerTrack v2 was not evaluated because access to its gated annotations was unavailable.

The three-reference variant and the pruned VGG variants were compared on validation only (3,183 queries). The ResNet baseline reached 84.26%; the same weights with three references reached 86.40%, and the three-reference-trained model reached 88.34%. Compact VGG reached 63.87% with SNIP and 59.00% with matched layer-random pruning; the wider layer-random VGG reached 67.39%. These are not test scores.

### Thirty paired draws on the same test queries

Each one-reference draw is nested within its corresponding six-reference draw. For an outfield class, the samples are different crops of one reference athlete, not six different athletes. SD is across random reference choices on **the same three matches**, not across new matches.

| Condition | 1 ref accuracy ± SD | 6 refs accuracy ± SD |
|---|---:|---:|
| ResNet-18 trained with one reference | 95.82% ± 1.28 pp | 97.15% ± 0.37 pp |
| ResNet-18 trained with six references | 95.72% ± 1.04 pp | 96.78% ± 0.26 pp |
| Frozen ImageNet ResNet-18 | 54.60% ± 5.96 pp | 72.30% ± 2.69 pp |
| HSV torso color | 78.53% ± 5.70 pp | 85.24% ± 4.56 pp |
| Frozen Market-1501 OSNet | 60.63% ± 5.62 pp | 79.15% ± 3.82 pp |

The repeated-draw evaluation reproduced the fixed six-reference confusion matrix. The separate transductive color clustering comparator could see all unlabelled queries together; it is not equivalent to independent crop inference. No comparison is claimed against a sports-specific ReID checkpoint of unknown match provenance.

### Prospective replay

The first 30% of available labeled timestamps define the bootstrap period. Up to 192 uniformly sampled bootstrap frames and 96 uniformly sampled later frames are considered per half. Both arms covered **3 of 6 halves**, since the other halves had no usable labeled bootstrap crop for at least one goalkeeper class. The table scores **the same 3,688 later boxes** in the three covered halves; it excludes the uncovered halves and does not establish an operational coverage rate for other matches or feeds.

| Checkpoint | 1 early reference per class | 6 early references per class |
|---|---:|---:|
| Trained with one reference | 96.29% | 97.02% |
| Trained with six references | 94.69% | 95.47% |

### Error analysis

On the fixed one-reference test, 117 boxes were wrong. In the 1,704 boxes with no annotated human-box overlap, 53 were wrong (3.11%); 47 of those crossed the outfield/goalkeeper role boundary. In the 128 boxes with at least 60% intersection coverage, 27 were wrong (21.09%). This is a box-overlap proxy, not proof of physical occlusion or the cause of each error. No threshold was selected on test. `SNGS-131` had 0 disagreements on 123 scored boxes after the role-specific audit and retraining.

## Apple M4 Pro deployment

FP16 PyTorch preserved all predictions in the separate precision replay and halved checkpoint size. Core ML INT8 quantizes weights and activations; calibration crops came only from train games 4, 6, and 9. Compute-plan placements are preferred devices, not measured per-operation execution traces. Latency is for a batch of **20 prepared crops**, not video decoding, detection, or tracking.

| Runtime | Accuracy | Median per 20 crops |
|---|---:|---:|
| GPU PyTorch FP32 | 96.77% | 6.17 ms |
| GPU PyTorch FP16 | 96.77% | 5.90 ms |
| CPU Core ML INT8 | 96.77% | 7.12 ms |
| GPU Core ML INT8 | 96.73% | 2.99 ms |
| Neural Engine Core ML FP16 | 96.77% | 2.55 ms |
| Neural Engine Core ML INT8 | 96.69% | 2.24 ms |

The regenerated twelve-second video was checked as a 150-frame 1280×824 MP4; its 2,443 boxes are an illustrative window, not the test set. See [`results/`](results/) for metrics and audit receipts, [`models/SHA256SUMS`](models/SHA256SUMS) for checkpoint hashes, and [`media/`](media/) for the demonstration.

## Comparison with the official SoccerNet baseline

<!-- PRTREID_COMPARISON_DETAIL -->
