# Results

| File | Contents | Producing script |
|---|---|---|
| `audit.json` | Prepared data, class mapping, exclusions, and label-normalization audit | `prepare.py` |
| `frozen-encoders.json` | Frozen SigLIP and CLIP with the same six references, and the Roboflow team pipeline with predicted and true roles, on six held-out matches | `experiments/frozen_encoders.py` |
| `formats.json` | Runtime environment and format accuracy/latency measurements | `evaluate_formats.py` |
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
