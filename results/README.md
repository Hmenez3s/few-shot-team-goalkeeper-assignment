# Results

| File | Contents | Producing script |
|---|---|---|
| `audit.json` | Prepared data, class mapping, exclusions, and label-normalization audit | `prepare.py` |
| `formats.json` | Runtime environment and format accuracy/latency measurements | `evaluate_formats.py` |
| `occlusion.json` | Error counts and confusion matrices by box-overlap proxy | `experiments/analyze_errors.py` |
| `prospective.json` | Prospective replay protocol, coverage, and scores | `experiments/prospective_evaluation.py` |
| `prtreid-comparison.json` | PRTReID clustering and reference-classifier comparison | `experiments/prtreid_comparison.py` |
| `research-reference-draws.json` | Paired reference-draw results, assignments, and uncertainty | `experiments/research_references.py` |
| `research-seeds-and-folds.json` | Training-seed summaries and held-out-game fold results | Seed runs: `train.py`; fold-analysis producer not retained |
| `result.json` | One-reference training configuration, validation, and test metrics | `train.py` |
| `six-references.json` | Six-reference training configuration, history, validation, and test metrics | `six_references.py` |
| `six-references.log` | Six-reference training and validation log | `six_references.py` |
| `training.log` | Data preparation and one-reference training log | `run.sh` |
| `verification.json` | Checkpoint hashes, test-box count, and prepared-split checks | Verification run; producer script not retained |
