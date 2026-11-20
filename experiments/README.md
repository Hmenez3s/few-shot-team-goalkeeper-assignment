# Experiments

Each script answers one question from [RESEARCH.md](../RESEARCH.md). Run them from the repository root as modules, for example `python -m experiments.analyze_errors --help`. `DATA` is a prepared data root (the `data/` folder after `./run.sh`).

| Script | Question | Writes |
|---|---|---|
| `analyze_errors` | Where do the errors happen, and how much does box overlap matter? | `DATA/work/corrected-error-analysis-v1.json` |
| `prospective_evaluation` | How well do references picked early in a half classify everything after them? | `DATA/work/prospective-evaluation-v1.json` |
| `research_references` | How much does the choice of reference crops matter, and how do colour and re-identification baselines compare? | file given by `--out` |
| `prtreid_comparison`, `prtreid_embed`, `tracklet_comparison` | How does the classifier compare with the official SoccerNet team method? | files given by `--out` |
| `frozen_encoders` | Do general-purpose encoders (SigLIP, CLIP) or the Roboflow team pipeline match the trained encoder? | file given by `--out` |
| `speed_general_encoders` | How much compute do our encoder, SigLIP and CLIP need per crop? | file given by `--out` |
| `export_eval_bundle` | Packs the fixed test sessions and calibration crops into one portable file | file given by `--out` |
| `gpu_formats` | How accurate and fast is the encoder in TensorRT FP16, INT8, FP8 and 2:4 sparsity on NVIDIA GPUs? | file given by `--out` |
| `sparse24` | Does the encoder keep its accuracy with 2:4 structured sparsity? | `--out`/result.json |
| `rotated_fold.sh` | Builds a fold where other matches are held out | a new data root |
| `summarize_seeds_and_folds` | How much do results move across training seeds and held-out matches? | file given by `--out` |
| `verify_split` | Do the checkpoints match, and does any identical crop appear in two splits? | file given by `--out` |
| `speed_comparison` | How much compute do our encoder and PRTReID need per crop? | file given by `--out` |

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

## Seeds and folds

`experiments/rotated_fold.sh "$DATA" seed-13 0` makes a copy of the data root with the same split (shift 0); train it with `python train.py --root "$PWD/seed-13" --seed 13`, and likewise for seeds 29, 42 and 57. Folds use shifts 1 and 2. Then:

```bash
python -m experiments.summarize_seeds_and_folds --official "$DATA" --seed-roots seed-13 seed-29 seed-42 seed-57 \
    --fold-roots fold-1 fold-2 --out results/research-seeds-and-folds.json
```
