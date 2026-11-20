import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import reference_bank
import train

SEED = 25
EPOCHS = 8
STEPS = 180
REFERENCES = 6
QUALITY_OVERLAP = 0.75
SELECTOR_SEED = SEED + 100
REPLACEMENT_SEED = SEED + 400
EVALUATION_SEED = SEED + 500


def load_bank(path, root, manifest, entry):
    expected = reference_bank.bank_identity(root, manifest, entry, path)
    with np.load(path, allow_pickle=False) as data:
        required = ("bank_schema", "protocol_id", "prepared_sha256", "cutoff_ms")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path}: missing bank identity fields: {', '.join(missing)}")
        identity = {}
        for key in required:
            value = data[key]
            if value.shape != ():
                raise ValueError(f"{path}: {key} must be a scalar")
            if key in ("protocol_id", "prepared_sha256"):
                if value.dtype.kind not in ("U", "S") or value.dtype.itemsize > reference_bank.BANK_TEXT_SIZE * 4:
                    raise ValueError(f"{path}: {key} must be a bounded text scalar")
                text = value.item()
                if len(text) != reference_bank.BANK_TEXT_SIZE:
                    raise ValueError(f"{path}: {key} must contain {reference_bank.BANK_TEXT_SIZE} characters")
                identity[key] = text.decode("ascii") if isinstance(text, bytes) else text
            else:
                if not np.issubdtype(value.dtype, np.integer):
                    raise ValueError(f"{path}: {key} must be an integer scalar")
                identity[key] = value.item()
        bank = {key: data[key] for key in ("x", "t", "y", "track", "desc")}
    for key, value in expected.items():
        if identity[key] != value:
            raise ValueError(f"{path}: {key} does not match the current unit manifest and prepared clip")
    n = len(bank["y"])
    if bank["x"].shape != (n, 96, 48, 3) or bank["x"].dtype != np.uint8:
        raise ValueError(f"{path}: x must be uint8 [N,96,48,3]")
    if any(bank[key].shape != (n,) for key in ("t", "y", "track")) or bank["desc"].shape != (n, 5):
        raise ValueError(f"{path}: expected t/y/track [N] and desc [N,5]")
    if not np.issubdtype(bank["t"].dtype, np.integer) or not np.issubdtype(bank["track"].dtype, np.integer):
        raise ValueError(f"{path}: t and track must be integer arrays")
    if np.any((bank["y"] < 0) | (bank["y"] >= 4)) or np.any(bank["t"] >= expected["cutoff_ms"]):
        raise ValueError(f"{path}: invalid labels or a bank timestamp is at/after the unit cutoff")
    if not np.isfinite(bank["desc"][:, 2:]).all():
        raise ValueError(f"{path}: non-pitch descriptors must be finite")
    pitch = bank["desc"][:, :2]
    missing_pitch = np.isnan(pitch).all(axis=1)
    present_pitch = np.isfinite(pitch).all(axis=1) & np.all((pitch >= 0) & (pitch <= 1), axis=1)
    if not np.all(missing_pitch | present_pitch):
        raise ValueError(f"{path}: pitch descriptors must be both NaN or both finite in [0,1]")
    return bank


def candidates_for(bank, cls, anchor_time, track, query_start):
    mask = (bank["y"] == cls) & (bank["t"] < query_start) & (bank["t"] != anchor_time)
    candidates = np.flatnonzero(mask)
    if cls < 2:
        candidates = candidates[bank["track"][candidates] == track]
    by_time = {}
    for index in candidates:
        by_time.setdefault(int(bank["t"][index]), int(index))
    candidates = list(by_time.values())
    clear = [index for index in candidates if bank["desc"][index, 3] <= QUALITY_OVERLAP]
    return clear if len(clear) >= REFERENCES - 1 else candidates


def resolve_episode(clip, bank, episode, session_id="session"):
    anchors, query = episode
    anchors = list(anchors)
    query_start = int(clip["t"][query].min())
    for cls, anchor in enumerate(anchors):
        def has_references(candidate):
            track = int(clip["track"][candidate])
            if cls < 2 and any(int(clip["track"][q]) == track for q in query if int(clip["y"][q]) == cls):
                return False
            return len(candidates_for(bank, cls, int(clip["t"][candidate]), track, query_start)) >= REFERENCES - 1

        if has_references(anchor):
            continue
        alternatives = train.usable_indices(clip, cls, True)
        if cls < 2:
            alternatives = alternatives[clip["track"][alternatives] >= 0]
        replacement = next((int(candidate) for candidate in alternatives
                            if int(candidate) != anchor and has_references(int(candidate))), None)
        if replacement is None:
            raise ValueError(f"{session_id}: class {cls} has no anchor with {REFERENCES - 1} eligible reference frames; no comparison produced")
        anchors[cls] = replacement
    return anchors, query

def make_session(clip, bank, episode, rng, strict):
    anchors, query = episode
    query_start = int(clip["t"][query].min())
    extras = []
    for cls, anchor in enumerate(anchors):
        candidates = candidates_for(bank, cls, int(clip["t"][anchor]), int(clip["track"][anchor]), query_start)
        if len(candidates) < REFERENCES - 1:
            if strict:
                raise ValueError(f"session cannot supply {REFERENCES} references per class before {query_start} ms")
            return None
        extras.append(rng.sample(candidates, REFERENCES - 1))
    support = []
    for cls, anchor in enumerate(anchors):
        extra = extras[cls]
        times = [int(clip["t"][anchor]), *map(int, bank["t"][extra])]
        if len(set(times)) != REFERENCES or max(times) >= query_start:
            raise ValueError("references must come from distinct frames before every query")
        if not np.all(bank["y"][extra] == cls):
            raise ValueError("reference class mismatch")
        track = int(clip["track"][anchor])
        if cls < 2 and not np.all(bank["track"][extra] == track):
            raise ValueError("outfield references must show the same athlete")
        if cls < 2 and any(int(clip["track"][q]) == track for q in query if int(clip["y"][q]) == cls):
            raise ValueError("reference athlete appears among the boxes to classify")
        support.extend([clip["x"][anchor], *[bank["x"][i] for i in extra]])
    return support, query


def batch_crops(crops, device):
    images = torch.from_numpy(np.asarray(crops).copy()).permute(0, 3, 1, 2).contiguous().float().div_(255)
    return ((images - train.MEAN) / train.STD).to(device)


def class_prototypes(features, per_class):
    return F.normalize(features, dim=1).reshape(4, per_class, -1).mean(dim=1)


def predict(model, clip, support, query, device):
    model.eval()
    with torch.inference_mode():
        prototypes = class_prototypes(model(batch_crops(support, device)), REFERENCES)
        output = []
        for start in range(0, len(query), 64):
            features = model(train.batch_images(clip, query[start:start + 64], device))
            output.extend(train.cosine_logits(features, prototypes).argmax(dim=1).cpu().tolist())
    return np.asarray(output, dtype=np.int64)


def sessions_for(entries, cache, bank_dir):
    root = bank_dir.parent.parent
    manifest, _, _ = train.load_manifest(root)
    sessions = []
    for entry in entries:
        clip = cache.get(entry["path"])
        episode = train.fixed_episode(clip)
        if episode is None:
            raise ValueError(f"no fixed episode for {entry['clip']}")
        bank_path = bank_dir / f"{entry['clip']}.npz"
        bank = load_bank(bank_path, root, manifest, entry)
        episode = resolve_episode(clip, bank, episode, entry["clip"])
        sessions.append((entry, clip, bank, episode))
    return sessions


def evaluate(model, sessions, device, label):
    rng = random.Random(EVALUATION_SEED)
    records = []
    for entry, clip, bank, episode in sessions:
        support, query = make_session(clip, bank, episode, rng, True)
        records.append({"game": entry["game"], "clip": entry["clip"], "truth": clip["y"][query],
                        "pred": predict(model, clip, support, query, device)})
        print(f"{label}: {entry['game']}/{entry['clip']} boxes={len(query)}", flush=True)
    return train.statistics(records)


def training_schedule(entries, cache):
    seed_rng = random.Random(SEED)
    eligible = [(entry, cache.get(entry["path"])) for entry in entries]
    for entry, clip in eligible:
        if train.choose_training_episode(clip, seed_rng) is None:
            raise ValueError(f"no training episode for {entry['clip']}")
    schedule = []
    for _ in range(EPOCHS):
        epoch = []
        for _ in range(STEPS):
            entry, clip = seed_rng.choice(eligible)
            epoch.append((entry, train.choose_training_episode(clip, seed_rng)))
        schedule.append(epoch)
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest, splits, games = train.load_manifest(root)
    bank_dir = root / "work" / "references"
    output = root / "work" / "six-references"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    cache = train.ClipCache(capacity=12)
    validation = sessions_for(splits["valid"], cache, bank_dir)
    banks = {entry["clip"]: load_bank(bank_dir / f"{entry['clip']}.npz", root, manifest, entry) for entry in splits["train"]}
    schedule = training_schedule(splits["train"], cache)
    model = train.build_model(root / "inputs" / "resnet18.pth", device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * STEPS)
    selector_rng = random.Random(SELECTOR_SEED)
    replacement_rng = random.Random(REPLACEMENT_SEED)
    checkpoint = output / "checkpoint.pth"
    history, best, resampled = [], None, 0
    print(f"starting {EPOCHS} epochs x {STEPS} episodes with {REFERENCES} references per class on {device}", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
        losses = []
        for entry, episode in schedule[epoch]:
            clip = cache.get(entry["path"])
            bank = banks[entry["clip"]]
            session = make_session(clip, bank, episode, selector_rng, False)
            if session is None:
                resampled += 1
                for _ in range(100):
                    episode = train.choose_training_episode(clip, replacement_rng)
                    if episode is not None and make_session(clip, bank, episode, selector_rng, False) is not None:
                        break
                session = make_session(clip, bank, episode, selector_rng, True)
            support, query = session
            prototypes = class_prototypes(model(batch_crops(support, device)), REFERENCES)
            logits = train.cosine_logits(model(train.batch_images(clip, query, device)), prototypes)
            loss = F.cross_entropy(logits, torch.as_tensor(clip["y"][query].copy(), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation, device, f"validation epoch {epoch + 1}")
        print(f"epoch {epoch + 1}/{EPOCHS} loss={np.mean(losses):.4f} validation accuracy={metrics['accuracy']:.6f} class mean={metrics['macro_class_accuracy']:.6f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": metrics})
        if best is None or metrics["macro_class_accuracy"] > best["validation"]["macro_class_accuracy"]:
            best = history[-1]
            pending = checkpoint.with_suffix(".pending")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "references_per_class": REFERENCES}, pending)
            pending.replace(checkpoint)
    locked = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(locked["model"])
    print(f"validation checkpoint locked at epoch {locked['epoch']}; opening test split", flush=True)
    test = evaluate(model, sessions_for(splits["test"], cache, bank_dir), device, "test")
    train.atomic_json(output / "result.json", {
        "device": str(device), "seed": SEED, "epochs": EPOCHS, "steps_per_epoch": STEPS,
        "references_per_class": REFERENCES, "selection": "anchor reference plus five random earlier frames per class",
        "quality_overlap_threshold": QUALITY_OVERLAP, "best_epoch": locked["epoch"], "resampled_episodes": resampled,
        "validation": best["validation"], "history": history, "test": test,
        "split_identity": {name: sorted(values) for name, values in games.items()},
    })
    print(f"result written: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
