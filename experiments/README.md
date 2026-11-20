# Experiments

Run these modules from the repository root. Replace bracketed paths with the relevant data, checkpoint, or intermediate file. `prtreid_embed` runs in a separate Python 3.9 environment with `prtreid` installed.

| Script | Question it answers | Command | Results file written |
|---|---|---|---|
| `analyze_errors` | Where do classification errors occur, including by box overlap? | `python -m experiments.analyze_errors --root [data-root]` | `[data-root]/work/corrected-error-analysis-v1.json` |
| `prospective_evaluation` | How well do early references classify later boxes? | `python -m experiments.prospective_evaluation --root [data-root] --checkpoint-one [one-checkpoint] --checkpoint-six [six-checkpoint]` | `[data-root]/work/prospective-evaluation-v1.json` |
| `research_references` | How do reference choice and appearance baselines compare on shared queries? | `python -m experiments.research_references --root [data-root] --bank [reference-bank] --one [one-checkpoint] --six [six-checkpoint] --out results/research-reference-draws.json` | `results/research-reference-draws.json` |
| `prtreid_comparison` | How does the reference classifier compare with PRTReID embeddings and clustering? | `python -m experiments.prtreid_comparison extract --root [data-root] --out [extract-directory]`; `python -m experiments.prtreid_comparison evaluate --extract [extract.npz] --embed [embeddings.npz] --out results/prtreid-comparison.json` | `results/prtreid-comparison.json` |
| `prtreid_embed` | What embeddings does the separate PRTReID environment produce for extracted crops? | `python -m experiments.prtreid_embed --extract [extract.npz] --out [embeddings.npz]` | `[embeddings.npz]` |
| `tracklet_comparison` | Does averaging predictions over a ground-truth tracklet change classification accuracy? | `python -m experiments.tracklet_comparison --root [data-root] --extract [extract.npz] --out [tracklet-results.json]` | `[tracklet-results.json]` |
