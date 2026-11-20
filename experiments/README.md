# Experiments

Each script answers one question from [RESEARCH.md](../RESEARCH.md). Run them from the repository root as modules, for example `python -m experiments.analyze_errors --help`. `DATA` is a prepared data root (the `data/` folder after `./run.sh`).

| Script | Question | Writes |
|---|---|---|
| `analyze_errors` | Where do the errors happen, and how much does box overlap matter? | `DATA/work/corrected-error-analysis-v1.json` |
| `prospective_evaluation` | How well do references picked early in a half classify everything after them? | `DATA/work/prospective-evaluation-v1.json` |
| `research_references` | How much does the choice of reference crops matter, and how do colour and re-identification baselines compare? | file given by `--out` |
| `prtreid_comparison`, `prtreid_embed`, `tracklet_comparison` | How does the classifier compare with the official SoccerNet team method? | files given by `--out` |
| `rotated_fold.sh` | Builds a fold where other matches are held out | a new data root |

## Comparison with the SoccerNet baseline

`prtreid_embed` runs in its own Python 3.9 environment with [PRTReID](https://github.com/VlSomers/prtreid) installed, because PRTReID needs `torch==1.13.1`. It also needs a checkout of [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) for the official configuration, and the official weights `prtreid-soccernet-baseline.pth.tar` and `hrnetv2_w32_imagenet_pretrained.pth` in one folder (`WEIGHTS`).

```bash
python -m experiments.prtreid_comparison extract --root "$DATA" --out run/extract
prtreid-env/bin/python -m experiments.prtreid_embed --extract run/extract/extract.npz --out run/embed.npz \
    --prtreid path/to/prtreid --sng path/to/sn-gamestate --weights "$WEIGHTS"
python -m experiments.prtreid_comparison evaluate --extract run/extract/extract.npz --embed run/embed.npz \
    --weights "$WEIGHTS" --out run/baseline.json
python -m experiments.tracklet_comparison --root "$DATA" --extract run/extract/extract.npz --out run/ours.json
```

For the second set of test matches, build a rotated fold and train both models on it first:

```bash
experiments/rotated_fold.sh "$DATA" fold
python train.py --root "$PWD/fold"
python six_references.py --root "$PWD/fold"
```

Then run the four commands above with `--root fold`, adding `--one fold/work/checkpoint.pth --six fold/work/six-references/checkpoint.pth --expected-one fold/work/result.json --expected-six fold/work/six-references/result.json` to `tracklet_comparison`. Both runs together give [`results/prtreid-comparison.json`](../results/prtreid-comparison.json).
