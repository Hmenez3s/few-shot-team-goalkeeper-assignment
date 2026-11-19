import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


CLASSES = ["player_left", "player_right", "goalkeeper_left", "goalkeeper_right"]
IMAGE_SIZE = (96, 48)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ClipCache:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, path):
        path = Path(path)
        if path not in self.items:
            with np.load(path, allow_pickle=False) as data:
                clip = {key: data[key] for key in ("x", "y", "t", "track", "early")}
            validate_clip(clip, path)
            self.items[path] = clip
            if len(self.items) > self.capacity:
                self.items.popitem(last=False)
        else:
            self.items.move_to_end(path)
        return self.items[path]


def validate_clip(clip, path):
    x, y, t, track, early = (clip[k] for k in ("x", "y", "t", "track", "early"))
    n = len(y)
    if x.shape != (n, 96, 48, 3) or x.dtype != np.uint8:
        raise ValueError(f"{path}: x must be uint8 [N,96,48,3]")
    if any(a.shape != (n,) for a in (y, t, track, early)):
        raise ValueError(f"{path}: y, t, track, early must be [N]")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= len(CLASSES))):
        raise ValueError(f"{path}: y contains invalid class indices")
    if not np.issubdtype(t.dtype, np.integer) or not np.issubdtype(track.dtype, np.integer):
        raise ValueError(f"{path}: t and track must be integer arrays")
    if not np.issubdtype(early.dtype, np.bool_):
        raise ValueError(f"{path}: early must be boolean")
    early_times = t[early]
    late_times = t[~early]
    if len(early_times) and len(late_times) and early_times.max() >= late_times.min():
        raise ValueError(f"{path}: early and late samples are not chronologically separated")


def load_manifest(root):
    manifest_path = root / "work" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("classes") != CLASSES:
        raise ValueError(f"manifest classes must equal {CLASSES}")
    splits = {name: [] for name in ("train", "valid", "test")}
    games = {name: set() for name in splits}
    seen_clips = set()
    work = (root / "work").resolve()
    for entry in manifest.get("clips", []):
        split, game, clip_id, relative = (entry.get(k) for k in ("split", "game", "clip", "path"))
        if split not in splits or not isinstance(game, str) or not game or not isinstance(clip_id, str) or not clip_id:
            raise ValueError("each clip needs a valid split, game, and clip id")
        key = (game, clip_id)
        if key in seen_clips:
            raise ValueError(f"duplicate clip: {key}")
        seen_clips.add(key)
        path = (root / relative).resolve()
        if not path.is_relative_to(work) or not path.is_file():
            raise ValueError(f"clip path must be an existing file inside work: {relative}")
        entry = {"split": split, "game": game, "clip": clip_id, "path": path}
        splits[split].append(entry)
        games[split].add(game)
    if any(not splits[name] for name in splits):
        raise ValueError("train, valid, and test splits must each be nonempty")
    if any(not games[name] for name in games) or any(games[a] & games[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("train, valid, and test game ID sets must be nonempty and pairwise disjoint")
    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("manifest audit must be an object")
    return manifest, splits, games


def usable_indices(clip, class_id, early):
    return np.flatnonzero((clip["y"] == class_id) & (clip["early"] == early))


def choose_training_episode(clip, rng):
    support = []
    query_candidates = []
    for class_id in range(4):
        refs = usable_indices(clip, class_id, True)
        later = usable_indices(clip, class_id, False)
        if len(refs) == 0 or len(later) == 0:
            return None
        if class_id < 2:
            refs = refs[clip["track"][refs] >= 0]
            later = later[clip["track"][later] >= 0]
            if not len(refs) or not len(later):
                return None
            later_tracks = set(map(int, clip["track"][later]))
            viable_refs = refs if len(later_tracks) > 1 else refs[clip["track"][refs] != next(iter(later_tracks))]
            if not len(viable_refs):
                return None
            ref = int(rng.choice(viable_refs))
            candidates = [int(q) for q in later if clip["track"][q] != clip["track"][ref]]
        else:
            ref = int(rng.choice(refs))
            candidates = list(map(int, later))
        if not candidates:
            return None
        support.append(ref)
        query_candidates.append(candidates)
    query_count = min(4, *(len(candidates) for candidates in query_candidates))
    query = [rng.sample(candidates, query_count) for candidates in query_candidates]
    return support, [i for group in query for i in group]


def fixed_episode(clip):
    support, query = [], []
    for class_id in range(4):
        refs = usable_indices(clip, class_id, True)
        later = usable_indices(clip, class_id, False)
        if not len(refs) or not len(later):
            return None
        if class_id < 2:
            refs = refs[clip["track"][refs] >= 0]
            later = later[clip["track"][later] >= 0]
        ref = None
        for candidate in refs:
            if class_id >= 2 or np.any(clip["track"][later] != clip["track"][candidate]):
                ref = int(candidate)
                break
        if ref is None:
            return None
        support.append(ref)
        choices = later if class_id >= 2 else later[clip["track"][later] != clip["track"][ref]]
        if not len(choices):
            return None
        query.extend(map(int, choices))
    return support, query


def build_model(weights_path, device):
    model = resnet18(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("ImageNet checkpoint must contain a state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("layer3.", "layer4.")))
    model.to(device)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def batch_images(clip, indices, device):
    images = torch.from_numpy(clip["x"][indices].copy()).permute(0, 3, 1, 2).contiguous().float().div_(255)
    images = (images - MEAN) / STD
    return images.to(device)


def cosine_logits(features, weights):
    return F.normalize(features, dim=1) @ F.normalize(weights, dim=1).T * 20


def train_episode(model, clip, episode, optimizer, device):
    support, query = episode
    support_labels = torch.as_tensor(clip["y"][support].copy(), dtype=torch.long, device=device)
    query_labels = torch.as_tensor(clip["y"][query].copy(), dtype=torch.long, device=device)
    features = model(batch_images(clip, support + query, device))
    support_features = features[:len(support)]
    query_features = features[len(support):]
    prototypes = torch.stack([support_features[support_labels == class_id].mean(dim=0) for class_id in range(4)])
    loss = F.cross_entropy(cosine_logits(query_features, prototypes), query_labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def predictions(model, clip, episode, device, batch_size=64):
    support, query = episode
    model.eval()
    with torch.no_grad():
        support_features = model(batch_images(clip, support, device))
        support_labels = clip["y"][support]
        prototypes = torch.stack([support_features[torch.as_tensor(support_labels == class_id, device=device)].mean(dim=0) for class_id in range(4)])
        output = []
        for start in range(0, len(query), batch_size):
            inds = query[start:start + batch_size]
            feats = model(batch_images(clip, inds, device))
            output.extend(cosine_logits(feats, prototypes).argmax(dim=1).cpu().tolist())
    return np.asarray(output, dtype=np.int64)


def frozen_predictions(model, clip, episode, device, batch_size=64):
    support, query = episode
    model.eval()
    with torch.no_grad():
        refs = F.normalize(model(batch_images(clip, support, device)), dim=1)
        output = []
        for start in range(0, len(query), batch_size):
            inds = query[start:start + batch_size]
            feats = F.normalize(model(batch_images(clip, inds, device)), dim=1)
            output.extend((feats @ refs.T).argmax(dim=1).cpu().tolist())
    return np.asarray(output, dtype=np.int64)


def statistics(records):
    matrix = np.zeros((4, 4), dtype=np.int64)
    per_clip = defaultdict(list)
    per_game = defaultdict(list)
    for row in records:
        y, pred = row["truth"], row["pred"]
        np.add.at(matrix, (y, pred), 1)
        per_clip[row["game"] + "\0" + row["clip"]].append((y, pred))
        per_game[row["game"]].append((y, pred))

    def summarize(rows):
        if not rows:
            return {"count": 0, "accuracy": 0.0, "macro_class_accuracy": 0.0, "class_recall": [0.0] * 4}
        ys = np.concatenate([pair[0] for pair in rows])
        ps = np.concatenate([pair[1] for pair in rows])
        cm = np.zeros((4, 4), dtype=np.int64)
        np.add.at(cm, (ys, ps), 1)
        denom = cm.sum(axis=1)
        recall = np.divide(cm.diagonal(), denom, out=np.zeros(4, dtype=float), where=denom > 0)
        return {"count": int(len(ys)), "accuracy": float((ys == ps).mean()), "macro_class_accuracy": float(recall.mean()), "class_recall": recall.tolist(), "confusion_matrix": cm.tolist()}

    base = summarize([(r["truth"], r["pred"]) for r in records])
    base["per_game_mean_accuracy"] = float(np.mean([summarize(v)["accuracy"] for v in per_game.values()])) if per_game else 0.0
    base["per_game"] = {game: summarize(values) for game, values in sorted(per_game.items())}
    base["per_clip"] = {key.replace("\0", "/"): summarize(values) for key, values in sorted(per_clip.items())}
    base["match_count"] = len(per_game)
    base["unit_count"] = len(per_clip)
    base["clip_count"] = len(per_clip)
    base["class_names"] = CLASSES
    return base


def evaluate(model, clips, cache, device, reference_model=None, label="evaluation"):
    trained, frozen = [], []
    sessions = 0
    last_log = time.monotonic()
    for entry in clips:
        clip = cache.get(entry["path"])
        episode = fixed_episode(clip)
        if episode is None:
            continue
        sessions += 1
        truth = np.asarray(clip["y"][episode[1]], dtype=np.int64)
        pred = predictions(model, clip, episode, device)
        base_pred = frozen_predictions(reference_model, clip, episode, device) if reference_model is not None else None
        trained.append({"game": entry["game"], "clip": entry["clip"], "truth": truth, "pred": pred})
        if base_pred is not None:
            frozen.append({"game": entry["game"], "clip": entry["clip"], "truth": truth, "pred": base_pred})
        now = time.monotonic()
        if now - last_log >= 60 or entry is clips[-1]:
            print(f"{label}: sessions={sessions}/{len(clips)}", flush=True)
            last_log = now
    trained_stats = statistics(trained)
    trained_stats["session_count"] = sessions
    trained_stats["manifest_unit_count"] = len(clips)
    frozen_stats = statistics(frozen) if reference_model is not None else None
    if frozen_stats is not None:
        frozen_stats["session_count"] = sessions
        frozen_stats["manifest_unit_count"] = len(clips)
    return trained_stats, frozen_stats


def source_clip_counts(audit):
    units = audit.get("units")
    if isinstance(units, list) and all(isinstance(unit, dict) and isinstance(unit.get("source_clip_ids"), list) for unit in units):
        counts = {split: set() for split in ("train", "valid", "test")}
        for unit in units:
            if unit.get("split") in counts:
                counts[unit["split"]].update(map(str, unit["source_clip_ids"]))
        return {split: len(values) for split, values in counts.items()}
    counts = audit.get("retained_clip_counts")
    if isinstance(counts, dict) and all(isinstance(counts.get(split), int) for split in ("train", "valid", "test")):
        return {split: counts[split] for split in ("train", "valid", "test")}
    return None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=180)
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("epochs and steps-per-epoch must be positive")
    root = args.root.resolve()
    manifest, splits, game_sets = load_manifest(root)
    device = torch.device("cpu" if args.smoke_cpu else "mps")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --smoke-cpu only for a smoke run")
    weights_path = root / "inputs" / "resnet18.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    print(f"starting {args.epochs} epochs x {args.steps_per_epoch} episodes on {device}", flush=True)
    cache = ClipCache()
    trainable = build_model(weights_path, device)
    frozen = build_model(weights_path, device)
    frozen.load_state_dict(trainable.state_dict())
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    for module in frozen.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    optimizer = torch.optim.AdamW((p for p in trainable.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    rng = random.Random(args.seed)
    usable = []
    for entry in splits["train"]:
        clip = cache.get(entry["path"])
        if choose_training_episode(clip, rng) is not None:
            usable.append(entry)
    if not usable:
        raise ValueError("no training clip can form a four-class early-support/later-query episode")
    train_episodes = 0
    valid_stats = None
    valid_frozen_stats = None
    best_score = -math.inf
    best_valid_stats = None
    best_valid_frozen_stats = None
    last_log = time.monotonic()
    for epoch in range(args.epochs):
        trainable.train()
        for module in trainable.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        losses = []
        for step in range(args.steps_per_epoch):
            entry = rng.choice(usable)
            clip = cache.get(entry["path"])
            episode = choose_training_episode(clip, rng)
            if episode is None:
                continue
            losses.append(train_episode(trainable, clip, episode, optimizer, device))
            train_episodes += 1
            scheduler.step()
            now = time.monotonic()
            if now - last_log >= 60 or step == args.steps_per_epoch - 1:
                print(f"epoch {epoch + 1}/{args.epochs} step {step + 1}/{args.steps_per_epoch} loss={np.mean(losses):.4f}", flush=True)
                last_log = now
        valid_stats, valid_frozen_stats = evaluate(trainable, splits["valid"], cache, device, frozen, f"validation epoch {epoch + 1}")
        if valid_stats["session_count"] == 0:
            raise ValueError("validation split has no eligible four-class support/query session")
        score = valid_stats["macro_class_accuracy"]
        print(f"validation epoch {epoch + 1}: macro_class_accuracy={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            checkpoint = root / "work" / "checkpoint.pth"
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save({"model": trainable.state_dict(), "epoch": epoch + 1, "validation_macro_class_accuracy": score}, temporary)
            temporary.replace(checkpoint)
            best_valid_stats = valid_stats
            best_valid_frozen_stats = valid_frozen_stats
    locked = torch.load(root / "work" / "checkpoint.pth", map_location=device, weights_only=True)
    trainable.load_state_dict(locked["model"])
    trainable.eval()
    print("validation checkpoint locked; opening test split", flush=True)
    test_stats, frozen_test_stats = evaluate(trainable, splits["test"], cache, device, frozen, "test")
    if test_stats["session_count"] == 0:
        raise ValueError("test split has no eligible four-class support/query session")
    source_counts = source_clip_counts(manifest["audit"])
    split_counts = {
        "train": {"episodes": train_episodes, "sessions": len(usable), "manifest_units": len(splits["train"])},
        "valid": {"sessions": best_valid_stats["session_count"], "manifest_units": len(splits["valid"])},
        "test": {"sessions": test_stats["session_count"], "manifest_units": len(splits["test"])},
    }
    if source_counts is not None:
        for split, count in source_counts.items():
            split_counts[split]["original_source_clips"] = count
    result = {
        "device": str(device), "smoke_cpu": args.smoke_cpu, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "checkpoint": "work/checkpoint.pth",
        "best_epoch": locked["epoch"], "validation": {"trained": best_valid_stats, "frozen_baseline": best_valid_frozen_stats},
        "test": {"trained": test_stats, "frozen_baseline": frozen_test_stats},
        "split_identity": {name: sorted(values) for name, values in game_sets.items()},
        "split_counts": split_counts,
        "audit": manifest["audit"],
    }
    atomic_json(root / "work" / "result.json", result)
    print(f"result written: {root / 'work' / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
