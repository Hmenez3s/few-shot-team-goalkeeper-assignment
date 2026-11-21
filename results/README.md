# Results

| File | Contents | Producing script |
|---|---|---|
| `audit.json` | Prepared data, class mapping, exclusions, and label-normalization audit | `prepare.py` |
| `bootstrap-intervals.json` | 95% cluster-bootstrap intervals (by tracklet and by match half) for every method on the six held-out matches, and for the differences to FewShot-R18 | `experiments/bootstrap_intervals.py` |
| `six-shot-variance.json` | Five training seeds and three rotated folds of the 6-shot model, each box alone and with tracklets | `six_references.py --seed`, `experiments/summarize_six_shot_variance.py` |
| `transfer-soccertrack.json` | Every encoder and baseline on SoccerTrack without retraining: 12 clips, two teams, 30 paired reference draws, reference athletes excluded from the queries | `experiments/transfer_datasets.py` (crops), `experiments/transfer_baselines.py` |
| `frozen-encoders.json` | Frozen SigLIP and CLIP with the same six references, and the Roboflow team pipeline with predicted and true roles, on six held-out matches | `experiments/frozen_encoders.py` |
| `formats.json` | Runtime environment and format accuracy/latency measurements | `evaluate_formats.py` |
| `gpu-formats-rtx3060.json`, `gpu-formats-rtx5070ti.json` | TensorRT FP16, INT8, FP8 and 2:4 sparse accuracy and latency, batch-256 throughput, and simulated INT4/NVFP4 accuracy per GPU | `experiments/gpu_formats.py` |
| `sparse24.json` | 2:4 pruning mask, accuracy before and after fine-tuning, selected epoch | `experiments/sparse24.py` |
| `occlusion.json` | Error counts and confusion matrices by box-overlap proxy | `experiments/analyze_errors.py` |
| `prospective.json` | Prospective replay protocol, coverage, and scores | `experiments/prospective_evaluation.py` |
| `prtreid-comparison.json` | Official SoccerNet team method against the reference classifier on six held-out matches: per-group accuracies, scores and confusion matrices | `experiments/prtreid_comparison.py`, `experiments/tracklet_comparison.py` |
| `research-reference-draws.json` | Paired reference-draw results, assignments, and uncertainty | `experiments/research_references.py` |
| `research-seeds-and-folds.json` | Training-seed summaries and held-out-game fold results | `experiments/summarize_seeds_and_folds.py` |
| `speed-comparison.json` | Parameters, multiply-accumulates per crop and CPU/GPU latency of our encoder and PRTReID, same machine and PyTorch | `experiments/speed_comparison.py` |
| `speed-general-encoders.json` | Parameters, operations per crop and CPU/GPU latency of our encoder, frozen SigLIP and CLIP, plus Roboflow's per-clip UMAP and K-means time | `experiments/speed_general_encoders.py` |
| `result.json` | One-reference training configuration, validation, and test metrics | `train.py` |
| `six-references.json` | Six-reference training configuration, history, validation, and test metrics | `six_references.py` |
| `six-references.log` | Six-reference training and validation log | `six_references.py` |
| `training.log` | Data preparation and one-reference training log | `run.sh` |
| `verification.json` | Checkpoint hashes, test-box count, and prepared-split checks | `experiments/verify_split.py` |
