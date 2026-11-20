import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18

import six_references
import train

PUBLISHED_SEED = 525
BATCH_SIZE = 64


def load_bank(path):
    bank = six_references.load_bank(path)
    if not np.issubdtype(bank["t"].dtype, np.integer) or not np.issubdtype(bank["track"].dtype, np.integer):
        raise ValueError(f"{path}: t and track must be integer arrays")
    return bank


def choose_support(clip, bank, episode, rng):
    replay = random.Random()
    replay.setstate(rng.getstate())
    support, _ = six_references.make_session(clip, bank, episode, rng, True)
    query = episode[1]
    query_start = int(clip["t"][query].min())
    chosen = []
    for cls, anchor in enumerate(episode[0]):
        candidates = six_references.candidates_for(
            bank, cls, int(clip["t"][anchor]), int(clip["track"][anchor]), query_start
        )
        extras = replay.sample(candidates, six_references.REFERENCES - 1)
        chosen.append([("clip", int(anchor)), *(('bank', int(i)) for i in extras)])
    return support, chosen



def encode(model, crops, device, batch_size=BATCH_SIZE):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = six_references.batch_crops(crops[start:start + batch_size], device)
            output.append(model(batch).detach().cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 512), dtype=np.float32)


def load_model(path, device):
    model = resnet18(weights=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    has_classifier = any(key.startswith("fc.") for key in state)
    if not has_classifier:
        model.fc = torch.nn.Identity()
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    return model


def predict_vectors(support, queries, refs_per_class):
    refs = F.normalize(torch.as_tensor(support, dtype=torch.float32), dim=1)
    prototypes = refs.reshape(4, refs_per_class, -1).mean(dim=1)
    query = F.normalize(torch.as_tensor(queries, dtype=torch.float32), dim=1)
    return (query @ F.normalize(prototypes, dim=1).T).argmax(dim=1).numpy()


def color_vector(crop):
    image = Image.fromarray(crop).convert("HSV")
    hsv = np.asarray(image)
    height, width = hsv.shape[:2]
    rgb = np.asarray(crop)
    content = np.argwhere(np.any(rgb != 0, axis=2))
    if len(content):
        top, left = content.min(axis=0)
        bottom, right = content.max(axis=0) + 1
    else:
        top, left, bottom, right = 0, 0, height, width
    content_height, content_width = bottom - top, right - left
    y0, y1 = top + int(content_height * 0.22), top + int(content_height * 0.68)
    x0, x1 = left + int(content_width * 0.25), left + int(content_width * 0.75)
    region = hsv[y0:y1, x0:x1]
    if not region.size:
        region = hsv[int(top):int(bottom), int(left):int(right)]
    h, s, v = region.reshape(-1, 3).astype(np.int32).T
    hue_bins = np.where(s < 32, 12, h * 12 // 256)
    bins = hue_bins * 16 + (s * 4 // 256) * 4 + v * 4 // 256
    counts = np.bincount(bins, minlength=208).astype(np.float32)
    return counts / max(1, counts.sum())


def color_predict(support, queries, refs_per_class, transductive=False):
    refs = np.asarray(support, dtype=np.float32).reshape(4, refs_per_class, -1)
    centers = refs.mean(axis=1)
    if transductive:
        for _ in range(20):
            labels = np.stack([((queries - center) ** 2).sum(axis=1) for center in centers], axis=1).argmin(axis=1)
            updated = centers.copy()
            for cls in range(4):
                members = queries[labels == cls]
                if len(members):
                    updated[cls] = (refs[cls].sum(axis=0) + members.sum(axis=0)) / (refs_per_class + len(members))
            if np.array_equal(updated, centers):
                break
            centers = updated
    return np.stack([((queries - center) ** 2).sum(axis=1) for center in centers], axis=1).argmin(axis=1)


def reid_encoder(path, device):
    if path is None:
        return None
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def encode_reid(model, crops, device):
    output = []
    with torch.inference_mode():
        for start in range(0, len(crops), BATCH_SIZE):
            batch = []
            for crop in crops[start:start + BATCH_SIZE]:
                resized = Image.fromarray(crop).resize((128, 256), Image.Resampling.BILINEAR)
                tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                batch.append((tensor - mean) / std)
            result = model(torch.stack(batch).to(device))
            if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.shape[0] != len(batch):
                raise ValueError("ReID TorchScript must return a [N,D] vector tensor")
            output.append(F.normalize(result.float(), dim=1).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 0), dtype=np.float32)


def statistics(records):
    return train.statistics([{"game": r["game"], "clip": r["clip"], "truth": r["truth"], "pred": r["pred"]} for r in records])


def repeated_record_sets(draw_count, include_reid):
    names = [f"trained_{checkpoint}_{references}"
             for checkpoint in ("one", "six") for references in ("one", "six")]
    names.extend(("frozen_one", "frozen_six", "color_anchored_one", "color_anchored_six",
                  "color_transductive_one", "color_transductive_six"))
    if include_reid:
        names.extend(("reid_one", "reid_six"))
    return {name: [[] for _ in range(draw_count)] for name in names}


def scores_for_draws(draw_records):
    result = {}
    for method, records in draw_records.items():
        metrics = [statistics(rows) for rows in records]
        result[method] = {
            "draws": len(metrics),
            "top1_accuracy": distribution([m["accuracy"] for m in metrics]),
            "macro_class_recall": distribution([m["macro_class_accuracy"] for m in metrics]),
            "confusion_matrices": [m["confusion_matrix"] for m in metrics],
            "per_match": [{game: {"accuracy": value["accuracy"], "macro_class_recall": value["macro_class_accuracy"]}
                           for game, value in metric["per_game"].items()} for metric in metrics],
        }
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Same-query reference and appearance baseline study")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--one", type=Path, required=True)
    parser.add_argument("--six", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reid-torchscript", type=Path)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    root, bank_dir = args.root.resolve(), args.bank.resolve()
    out = args.out.resolve()
    if out.exists():
        parser.error(f"--out must be a new run-owned path: {out}")
    manifest, splits, _ = train.load_manifest(root)
    entries = splits[args.split]
    if args.reid_torchscript is not None and not args.reid_torchscript.is_file():
        parser.error("--reid-torchscript must name an existing local TorchScript file")
    reid_available = args.reid_torchscript is not None
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    one_model, six_model = load_model(args.one, device), load_model(args.six, device)
    frozen_model = load_model(root / "inputs" / "resnet18.pth", device)
    reid_model = reid_encoder(args.reid_torchscript, device) if reid_available else None
    published_rng = random.Random(PUBLISHED_SEED)
    draw_rngs = [random.Random(PUBLISHED_SEED + draw + 1) for draw in range(args.draws)]
    nested_rngs = [random.Random(PUBLISHED_SEED * 1009 + draw) for draw in range(args.draws)]
    used_paths = [args.one.resolve(), args.six.resolve(), root / "inputs" / "resnet18.pth", root / "work" / "manifest.json"]
    if reid_available:
        used_paths.append(args.reid_torchscript.resolve())
    selected_inputs = {}

    records = {name: [] for name in ("fixed_original_one", "fixed_published_six")}
    repeated = repeated_record_sets(args.draws, reid_model is not None)
    for entry in entries:
        with np.load(entry["path"], allow_pickle=False) as data:
            clip = {key: data[key] for key in ("x", "y", "t", "track", "early")}
        train.validate_clip(clip, entry["path"])
        episode = train.fixed_episode(clip)
        if episode is None:
            raise ValueError(f"no fixed episode for {entry['clip']}")
        bank_path = bank_dir / f"{entry['clip']}.npz"
        bank = load_bank(bank_path)
        _, published_ids = choose_support(clip, bank, episode, published_rng)
        draw_selections = []
        for draw in range(args.draws):
            _, selected_ids = choose_support(clip, bank, episode, draw_rngs[draw])
            nested_ids = []
            for cls in range(4):
                pick = nested_rngs[draw].randrange(six_references.REFERENCES)
                nested_ids.append([selected_ids[cls][pick]])
            draw_selections.append((selected_ids, nested_ids))
        used_paths.extend([entry["path"], bank_path])
        selected_inputs[entry["game"] + "/" + entry["clip"]] = {
            "query_indices": list(map(int, episode[1])),
            "fixed_original": [[int(anchor)] for anchor in episode[0]],
            "fixed_published_six": [[[kind, index] for kind, index in group] for group in published_ids],
            "draws": [{"six": [[[kind, index] for kind, index in group] for group in selected],
                       "nested_one": [[[kind, index] for kind, index in group] for group in nested]}
                      for selected, nested in draw_selections],
        }
        query = episode[1]
        ids = {("clip", int(i)) for i in (*episode[0], *query)}
        for group in published_ids:
            ids.update(group)
        for selected, nested in draw_selections:
            for group in (*selected, *nested):
                ids.update(group)
        ordered = sorted(ids, key=lambda item: (item[0], item[1]))
        crops = [clip["x"][i] if kind == "clip" else bank["x"][i] for kind, i in ordered]
        positions = {key: pos for pos, key in enumerate(ordered)}
        one_features = encode(one_model, crops, device)
        six_features = encode(six_model, crops, device)
        frozen_features = encode(frozen_model, crops, device)
        colors = np.stack([color_vector(crop) for crop in crops])
        reid_features = encode_reid(reid_model, crops, device) if reid_model is not None else None
        query_pos = [positions[("clip", i)] for i in query]
        truth = np.asarray(clip["y"][query], dtype=np.int64)
        query_data = {"game": entry["game"], "clip": entry["clip"], "truth": truth}
        original_ids = [[("clip", int(anchor))] for anchor in episode[0]]
        for name, selection in (("fixed_original_one", original_ids), ("fixed_published_six", published_ids)):
            refs = [positions[key] for group in selection for key in group]
            size = len(selection[0])
            for model_name, features in (("trained_one", one_features), ("trained_six", six_features)):
                records[name].append({**query_data, "pred": predict_vectors(features[refs], features[query_pos], size), "model": model_name})
        for draw, (six_ids, nested_ids) in enumerate(draw_selections):
            for size, selection in ((1, nested_ids), (6, six_ids)):
                refs = [positions[key] for group in selection for key in group]
                for model_name, features in (("trained_one", one_features), ("trained_six", six_features), ("frozen", frozen_features)):
                    predictions = predict_vectors(features[refs], features[query_pos], size)
                    repeated[f"{model_name}_{'one' if size == 1 else 'six'}"][draw].append({**query_data, "pred": predictions})
                color_refs = colors[refs]
                for mode in ("anchored", "transductive"):
                    pred = color_predict(color_refs, colors[query_pos], size, transductive=mode == "transductive")
                    repeated[f"color_{mode}_{'one' if size == 1 else 'six'}"][draw].append({**query_data, "pred": pred})
                if reid_features is not None:
                    pred = predict_vectors(reid_features[refs], reid_features[query_pos], size)
                    repeated[f"reid_{'one' if size == 1 else 'six'}"][draw].append({**query_data, "pred": pred})
        del one_features, six_features, frozen_features, reid_features
        print(f"{args.split}: {entry['game']}/{entry['clip']} session {len(selected_inputs)}/{len(entries)} boxes={len(query)} crops={len(ordered)}", flush=True)

    repeated_stats = scores_for_draws(repeated)
    paired = {}
    baseline = repeated["trained_one_one"]
    for method, draws in repeated.items():
        if method == "trained_one_one":
            continue
        deltas, disagreements = [], []
        for base_rows, method_rows in zip(baseline, draws):
            base_predictions = np.concatenate([row["pred"] for row in base_rows])
            method_predictions = np.concatenate([row["pred"] for row in method_rows])
            truth = np.concatenate([row["truth"] for row in base_rows])
            deltas.append(float(np.mean(method_predictions == truth) - np.mean(base_predictions == truth)))
            disagreements.append(float(np.mean(method_predictions != base_predictions)))
        paired[method] = {"top1_delta_vs_trained_one": distribution(deltas),
                          "prediction_disagreement_rate": distribution(disagreements)}
    fixed_stats = {}
    for name, rows in records.items():
        fixed_stats[name] = {}
        for model_name in ("trained_one", "trained_six"):
            selected = [row for row in rows if row["model"] == model_name]
            fixed_stats[name][model_name] = statistics(selected)
    published = json.loads((Path(__file__).parent / "results" / "six-references.json").read_text())[args.split]
    observed = fixed_stats["fixed_published_six"]["trained_six"]["confusion_matrix"]
    fixed_stats["published_six_confusion_verification"] = {
        "expected": published["confusion_matrix"], "observed": observed, "matches": published["confusion_matrix"] == observed,
        "note": "Numerically tied argmax classes may differ across kernels; mismatch is reported without altering predictions."}
    fingerprint = {str(path): sha256(path) for path in sorted(set(p.resolve() for p in used_paths))}
    result = {
        "protocol": {"split": args.split, "draws": args.draws, "session_count": len(entries), "fixed_episode": "train.fixed_episode",
                     "reference_counts_per_class": [1, six_references.REFERENCES],
                     "published_six_seed": PUBLISHED_SEED, "quality_overlap_threshold": six_references.QUALITY_OVERLAP,
                     "candidate_protocol": "six_references.candidates_for / make_session; same earlier-frame and quality fallback rules",
                     "draw_protocol": "one six-reference selection per session and draw; one reference/class sampled nested from that six",
                     "query_identity": "identical fixed_episode query indices for all methods and draws",
                     "outfield_identity": "references restricted to anchor athlete; fixed_episode excludes that athlete from outfield queries",
                     "reid_input": "ImageNet-normalized RGB NCHW [N,3,256,128], vector output [N,D]" if reid_model else None,
                     "reid_status": "included" if reid_model else "omitted: optional TorchScript path was absent or did not exist",
                     "transductive": "unlabeled query color vectors iteratively assigned to reference-seeded prototypes and included in prototype updates; separately reported"},
        "fixed": fixed_stats,
        "repeated_draws": repeated_stats,
        "paired_assignments_vs_trained_one": paired,
        "uncertainty": f"{len({entry['game'] for entry in entries})} matches only; match-level variation is descriptive, crops are not independent inferential units.",
        "verification_fingerprint": {"sha256_inputs": fingerprint, "selected_inputs": selected_inputs},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
