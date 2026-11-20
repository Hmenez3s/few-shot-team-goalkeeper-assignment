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

The SoccerNet Game State Reconstruction baseline assigns teams in three steps ([sn-gamestate](https://github.com/SoccerNet/sn-gamestate) at `1c95834`): PRTReID predicts each box's role and embedding; roles are majority-voted per tracklet; the mean embeddings of player tracklets are split by `KMeans(n_clusters=2, random_state=0)` per clip, and each cluster takes the side of its mean pitch position, while each goalkeeper takes the side of its own pitch position. We ran this logic on the same boxes we score, with ground-truth tracklets and pitch coordinates instead of a detector, tracker and calibration. This removes upstream errors for both methods. Boxes the baseline cannot assign (a non-player role or a missing pitch position) count as wrong.

The official checkpoint (`prtreid-soccernet-baseline.pth.tar`, [Zenodo 10653453](https://zenodo.org/records/10653453)) was trained on SoccerNet-Tracking train, according to the PRTReID paper. Those 57 clips are exactly the SN-GSR-2025 train clips (games 4, 6, 9), so no evaluated match was seen by it. Our models for matches 7, 8, 11 were trained on 4, 6, 9 and selected on 2, 3, 5; for matches 2, 3, 5 they were retrained on 7, 8, 11 and selected on 4, 6, 9.

**Score** is the mean of the four per-group accuracies; **accuracy** counts every box equally. "Tracklets" averages our embeddings over each ground-truth tracklet before comparing them with the references, the same information the baseline uses.

**Matches 7, 8, 11 (2,659 boxes)**

| Method | Score | Accuracy | Team A | Team B | Goalkeeper A | Goalkeeper B |
|---|---:|---:|---:|---:|---:|---:|
| SoccerNet baseline, official (predicted roles, side from pitch position) | 83.55% | 97.29% | 99.37% | 98.95% | 76.62% | 59.26% |
| SoccerNet baseline, true roles | 96.65% | 94.25% | 94.55% | 93.27% | 100.00% | 98.77% |
| SoccerNet baseline, true roles and best cluster-to-team mapping (upper bound) | 99.31% | 99.25% | 99.37% | 99.11% | 100.00% | 98.77% |
| Ours, one reference, each box | 86.10% | 95.60% | 96.05% | 97.73% | 75.32% | 75.31% |
| Ours, one reference, tracklets | 88.99% | 98.16% | 98.82% | 100.00% | 81.82% | 75.31% |
| Ours, six references, each box | 93.81% | 96.77% | 96.45% | 97.89% | 88.31% | 92.59% |
| Ours, six references, tracklets | 98.98% | 99.70% | 99.61% | 100.00% | 100.00% | 96.30% |

**Matches 2, 3, 5 (3,183 boxes)**

| Method | Score | Accuracy | Team A | Team B | Goalkeeper A | Goalkeeper B |
|---|---:|---:|---:|---:|---:|---:|
| SoccerNet baseline, official (predicted roles, side from pitch position) | 50.93% | 53.25% | 48.42% | 58.42% | 38.46% | 58.43% |
| SoccerNet baseline, true roles | 89.41% | 83.47% | 75.15% | 90.18% | 92.31% | 100.00% |
| SoccerNet baseline, true roles and best cluster-to-team mapping (upper bound) | 89.41% | 83.47% | 75.15% | 90.18% | 92.31% | 100.00% |
| Ours, one reference, each box | 89.49% | 87.15% | 84.55% | 89.13% | 91.03% | 93.26% |
| Ours, one reference, tracklets | 93.64% | 90.89% | 85.16% | 95.81% | 93.59% | 100.00% |
| Ours, six references, each box | 91.34% | 90.07% | 88.45% | 91.36% | 92.31% | 93.26% |
| Ours, six references, tracklets | 96.08% | 96.83% | 94.56% | 99.21% | 96.15% | 94.38% |

**All six matches (5,842 boxes)**

| Method | Score | Accuracy | Team A | Team B | Goalkeeper A | Goalkeeper B |
|---|---:|---:|---:|---:|---:|---:|
| SoccerNet baseline, official (predicted roles, side from pitch position) | 66.15% | 73.30% | 71.84% | 76.53% | 57.42% | 58.82% |
| SoccerNet baseline, true roles | 92.79% | 88.38% | 84.07% | 91.56% | 96.13% | 99.41% |
| SoccerNet baseline, true roles and best cluster-to-team mapping (upper bound) | 94.00% | 90.65% | 86.28% | 94.17% | 96.13% | 99.41% |
| Ours, one reference, each box | 87.69% | 91.00% | 89.84% | 92.97% | 83.23% | 84.71% |
| Ours, one reference, tracklets | 91.27% | 94.20% | 91.44% | 97.68% | 87.74% | 88.24% |
| Ours, six references, each box | 92.42% | 93.12% | 92.13% | 94.28% | 90.32% | 92.94% |
| Ours, six references, tracklets | 97.45% | 98.13% | 96.88% | 99.57% | 98.06% | 95.29% |

In match 3 the baseline's role vote marks 418 of the 1,072 scored boxes as referees or the ball. With true roles its score on matches 2, 3, 5 rises to 89.41%; the remaining errors come from K-means splitting a team, since the best cluster-to-team mapping gives the same score. On matches 7, 8, 11 the pitch-side rule also swaps teams in some clips (96.65% with true roles, 99.31% with the best mapping). Even with true roles and the best possible cluster-to-team mapping, it scores 94.00% over six matches against 97.45% for six references with tracklets.

### General-purpose encoders and the Roboflow pipeline

Two more baselines use the same boxes, tracklets and pitch positions. First, our encoder is replaced by frozen SigLIP (base, patch 16, 224 px) or CLIP (ViT-B/32), keeping the same six reference crops per group, cosine prototypes and tracklet averaging. Second, the team pipeline of the [Roboflow sports toolkit](https://github.com/roboflow/sports) at commit `42c80c0`: per clip, the mean SigLIP feature of each player tracklet is reduced with UMAP (3 components) and split by K-means into two clusters that label every box; clusters take the side of their mean pitch position and each goalkeeper joins the team whose players are nearer along the pitch. That pipeline needs the role of each box. It is run with the roles voted from PRTReID's role head, and once more with the true roles. All encoders see the same 96×48 crops. Score is the mean of the four per-group accuracies.

| Method | 7, 8, 11 | 2, 3, 5 | All six | Accuracy (all six) |
|---|---:|---:|---:|---:|
| Roboflow pipeline, roles from PRTReID | 82.43% | 60.45% | 70.85% | 82.68% |
| Roboflow pipeline, true roles (not directly comparable) | 98.15% | 94.13% | 96.05% | 95.31% |
| Frozen CLIP, six references, each box | 68.22% | 60.14% | 63.83% | 66.38% |
| Frozen CLIP, six references, tracklets | 88.47% | 76.74% | 82.17% | 83.11% |
| Frozen SigLIP, six references, each box | 81.70% | 67.80% | 74.34% | 76.67% |
| Frozen SigLIP, six references, tracklets | 93.09% | 83.52% | 87.98% | 89.08% |

With the same six references and tracklets, the trained encoder scores 97.45% against 87.98% for frozen SigLIP. With predicted roles, the Roboflow pipeline improves on the SoccerNet baseline for outfield players but not for goalkeepers (57.42% and 57.65%), since both use the same role predictions. Given the true roles it reaches 96.05%. That variant receives the part of the task our model has to solve, telling goalkeepers from outfield players, so it is not directly comparable; our model with tracklets still scores higher. Numbers: [`results/frozen-encoders.json`](results/frozen-encoders.json); code: [`experiments/frozen_encoders.py`](experiments/frozen_encoders.py).

Reproduce: build the rotated fold with [`experiments/rotated_fold.sh`](experiments/rotated_fold.sh), train both models on it with `train.py` and `six_references.py`, then run the commands in [`experiments/README.md`](experiments/README.md). Numbers: [`results/prtreid-comparison.json`](results/prtreid-comparison.json).
