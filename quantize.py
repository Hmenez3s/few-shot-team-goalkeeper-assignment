import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

import six_references
import train

BATCH = 20
CALIBRATION_SEED = 2025
CALIBRATION_PER_CLASS = 10


def load_float_model(root, checkpoint):
    model = train.build_model(root / "inputs" / "resnet18.pth", torch.device("cpu"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state


def calibration_batches(root):
    _, splits, games = train.load_manifest(root)
    rng = np.random.default_rng(CALIBRATION_SEED)
    batches = []
    for entry in splits["train"]:
        clip = train.ClipCache(1).get(entry["path"])
        picks = [rng.choice(np.flatnonzero(clip["y"] == cls), CALIBRATION_PER_CLASS, replace=False) for cls in range(4)]
        crops = clip["x"][np.stack(picks, axis=1).reshape(-1)]
        tensor = six_references.batch_crops(crops, torch.device("cpu")).numpy()
        batches.extend({"crops": np.ascontiguousarray(tensor[i:i + BATCH])} for i in range(0, len(tensor), BATCH))
    if any(len(batch["crops"]) != BATCH for batch in batches):
        raise ValueError("calibration crops must fill whole batches")
    return batches, sorted(games["train"])


def export_fp16(state, destination):
    state = dict(state)
    state["model"] = {key: value.half() if value.is_floating_point() else value for key, value in state["model"].items()}
    torch.save(state, destination)


def export_int8(model, root, destination):
    import coremltools as ct
    from coremltools.optimize import coreml as optimize
    from coremltools.optimize.coreml.experimental import OpActivationLinearQuantizerConfig, linear_quantize_activations

    with torch.no_grad():
        traced = torch.jit.trace(model, torch.zeros(BATCH, 3, 96, 48), check_trace=False)
    float16 = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name="crops", shape=(BATCH, 3, 96, 48), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    batches, games = calibration_batches(root)
    activations = optimize.OptimizationConfig(global_config=OpActivationLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    weights = optimize.OptimizationConfig(global_config=optimize.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    quantized = optimize.linear_quantize_weights(linear_quantize_activations(float16, activations, batches), weights)
    quantized.short_description = "Six-reference ResNet-18 encoder, INT8 weights and activations. Input: 20 normalised 96x48 crops. Output: 512-number embeddings."
    quantized.save(str(destination))
    return {"calibration_matches": games, "calibration_crops": len(batches) * BATCH, "calibration_seed": CALIBRATION_SEED}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/six-references.pth"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Torch version")
    root = args.root.resolve()
    model, state = load_float_model(root, args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    export_fp16(state, args.out / "six-references-fp16.pth")
    print(f"wrote {args.out / 'six-references-fp16.pth'}", flush=True)
    info = export_int8(model, root, args.out / "six-references-int8.mlpackage")
    print(f"wrote {args.out / 'six-references-int8.mlpackage'}: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
