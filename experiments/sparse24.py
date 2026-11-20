import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import six_references
import train

REPO = Path(__file__).resolve().parent.parent


def build_2_4_mask(weight):
    if weight.ndim != 4 or weight.shape[1] % 4:
        raise ValueError("2:4 masks require Conv2d weights with input channels divisible by four")
    output_channels, input_channels, height, width = weight.shape
    groups = weight.detach().abs().reshape(output_channels, input_channels // 4, 4, height, width)
    kept = groups.topk(2, dim=2).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(2, kept, True)
    return mask.reshape_as(weight)


def create_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels % 4 == 0:
            masks[name] = build_2_4_mask(module.weight)
    return masks


def apply_masks(model, masks):
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            modules[name].weight.mul_(mask)


def mask_statistics(model, masks):
    modules = dict(model.named_modules())
    eligible_layers = {}
    dense_layers = {}
    eligible_zero_count = 0
    eligible_weight_count = 0
    encoder_zero_count = 0
    encoder_weight_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        zero_count = int((module.weight == 0).sum().item())
        weight_count = module.weight.numel()
        encoder_zero_count += zero_count
        encoder_weight_count += weight_count
        fraction = zero_count / weight_count
        if name in masks:
            eligible_layers[name] = fraction
            eligible_zero_count += zero_count
            eligible_weight_count += weight_count
        else:
            dense_layers[name] = fraction
    return {
        "eligible_layers": eligible_layers,
        "dense_layers": dense_layers,
        "eligible_conv_sparsity": eligible_zero_count / eligible_weight_count,
        "encoder_conv_sparsity": encoder_zero_count / encoder_weight_count,
        "eligible_zero_weights": eligible_zero_count,
        "eligible_weights": eligible_weight_count,
        "encoder_zero_weights": encoder_zero_count,
        "encoder_weights": encoder_weight_count,
    }


def check_2_4(model, masks):
    modules = dict(model.named_modules())
    violations = []
    with torch.no_grad():
        for name in masks:
            weight = modules[name].weight
            groups = weight.reshape(weight.shape[0], weight.shape[1] // 4, 4, *weight.shape[2:])
            zero_counts = (groups == 0).sum(dim=2)
            if not bool((zero_counts >= 2).all()):
                violations.append(name)
    return {"passes": not violations, "violating_layers": violations}


def training_schedule(entries, cache, epochs, steps):
    seed_rng = random.Random(six_references.SEED)
    eligible = [(entry, cache.get(entry["path"])) for entry in entries]
    for entry, clip in eligible:
        if train.choose_training_episode(clip, seed_rng) is None:
            raise ValueError(f"no training episode for {entry['clip']}")
    schedule = []
    for _ in range(epochs):
        epoch = []
        for _ in range(steps):
            entry, clip = seed_rng.choice(eligible)
            epoch.append((entry, train.choose_training_episode(clip, seed_rng)))
        schedule.append(epoch)
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "models/six-references.pth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1:
        parser.error("epochs and steps must be positive")
    epochs, steps = (1, 2) if args.smoke else (args.epochs, args.steps)
    root = args.root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.out.resolve()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    random.seed(six_references.SEED)
    np.random.seed(six_references.SEED)
    torch.manual_seed(six_references.SEED)
    if device.type == "mps":
        torch.mps.manual_seed(six_references.SEED)

    manifest, splits, games = train.load_manifest(root)
    bank_dir = root / "work" / "references"
    cache = train.ClipCache(capacity=12)
    validation = six_references.sessions_for(splits["valid"], cache, bank_dir)
    test_sessions = six_references.sessions_for(splits["test"], cache, bank_dir)
    banks = {
        entry["clip"]: six_references.load_bank(bank_dir / f"{entry['clip']}.npz", root, manifest, entry)
        for entry in splits["train"]
    }
    schedule = training_schedule(splits["train"], cache, epochs, steps)

    model = train.build_model(root / "inputs" / "resnet18.pth", device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    masks = create_masks(model)
    apply_masks(model, masks)
    stats = mask_statistics(model, masks)
    validation_before = six_references.evaluate(model, validation, device, "validation before fine-tune")
    test_before = six_references.evaluate(model, test_sessions, device, "test before fine-tune")

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps)
    selector_rng = random.Random(six_references.SELECTOR_SEED)
    replacement_rng = random.Random(six_references.REPLACEMENT_SEED)
    history = []
    best = None
    resampled = 0
    module_map = dict(model.named_modules())
    for epoch in range(epochs):
        model.train()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        losses = []
        for entry, episode in schedule[epoch]:
            clip = cache.get(entry["path"])
            bank = banks[entry["clip"]]
            session = six_references.make_session(clip, bank, episode, selector_rng, False)
            if session is None:
                resampled += 1
                for _ in range(100):
                    episode = train.choose_training_episode(clip, replacement_rng)
                    if episode is not None and six_references.make_session(clip, bank, episode, selector_rng, False) is not None:
                        break
                session = six_references.make_session(clip, bank, episode, selector_rng, True)
            support, query = session
            prototypes = six_references.class_prototypes(
                model(six_references.batch_crops(support, device)), six_references.REFERENCES
            )
            logits = train.cosine_logits(model(train.batch_images(clip, query, device)), prototypes)
            loss = F.cross_entropy(
                logits,
                torch.as_tensor(clip["y"][query].copy(), dtype=torch.long, device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for name, mask in masks.items():
                    module_map[name].weight.mul_(mask)
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        metrics = six_references.evaluate(model, validation, device, f"validation epoch {epoch + 1}")
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": metrics}
        history.append(row)
        print(
            f"epoch {epoch + 1}/{epochs} loss={row['loss']:.4f} "
            f"validation macro recall={metrics['macro_class_accuracy']:.6f}",
            flush=True,
        )
        if best is None or metrics["macro_class_accuracy"] > best["validation"]["macro_class_accuracy"]:
            best = row
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "references_per_class": six_references.REFERENCES,
                 "sparsity": "2:4"},
                output / "checkpoint.pth",
            )

    selected = torch.load(output / "checkpoint.pth", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    test_after = six_references.evaluate(model, test_sessions, device, "test after fine-tune")
    check = check_2_4(model, masks)
    if not check["passes"]:
        raise RuntimeError(f"2:4 constraint failed: {check['violating_layers']}")
    result = {
        "device": str(device),
        "seed": six_references.SEED,
        "epochs": epochs,
        "steps_per_epoch": steps,
        "references_per_class": six_references.REFERENCES,
        "selection": "anchor reference plus five random earlier frames per class",
        "quality_overlap_threshold": six_references.QUALITY_OVERLAP,
        "resampled_episodes": resampled,
        "mask": stats,
        "validation_before_fine_tune": validation_before,
        "test_before_fine_tune": test_before,
        "history": history,
        "selected_epoch": selected["epoch"],
        "validation": best["validation"],
        "test": test_after,
        "two_of_four_check": check,
        "split_identity": {name: sorted(values) for name, values in games.items()},
    }
    train.atomic_json(output / "result.json", result)
    print(f"result written: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
