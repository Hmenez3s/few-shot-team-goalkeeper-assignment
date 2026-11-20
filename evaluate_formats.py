import argparse
import random
import statistics
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import six_references
import train

BATCH = 20
FORMATS = [
    ("CPU", "PyTorch", "FP32"),
    ("CPU", "Core ML", "INT8"),
    ("GPU", "PyTorch", "FP32"),
    ("GPU", "PyTorch", "FP16"),
    ("GPU", "Core ML", "INT8"),
    ("Neural Engine", "Core ML", "FP16"),
    ("Neural Engine", "Core ML", "INT8"),
]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class TorchEncoder:
    def __init__(self, root, checkpoint, device, dtype):
        self.device, self.dtype = torch.device(device), dtype
        self.model = train.build_model(root / "inputs" / "resnet18.pth", self.device).to(dtype=dtype)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        self.model.eval()
        self.sync = torch.mps.synchronize if self.device.type == "mps" else (lambda: None)

    def __call__(self, crops):
        with torch.inference_mode():
            output = self.model(six_references.batch_crops(crops, self.device).to(dtype=self.dtype)).float().cpu().numpy()
        self.sync()
        return output


class CoreMLEncoder:
    def __init__(self, package, compute_units):
        import coremltools as ct
        self.model = ct.models.MLModel(str(package), compute_units=compute_units)
        self.package, self.compute_units = package, compute_units

    def __call__(self, crops):
        data = six_references.batch_crops(crops, torch.device("cpu")).numpy()
        output = []
        for start in range(0, len(data), BATCH):
            chunk = data[start:start + BATCH]
            count = len(chunk)
            if count < BATCH:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], BATCH - count, axis=0)))
            output.append(np.asarray(self.model.predict({"crops": np.ascontiguousarray(chunk)})["embedding"], dtype=np.float32)[:count])
        return np.concatenate(output)

    def placement(self):
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(self.model.get_compiled_model_path(), self.compute_units)
        counts = Counter()
        for operation in plan.model_structure.program.functions["main"].block.operations:
            if operation.operator_name.rsplit(".", 1)[-1] == "conv":
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                counts[type(usage.preferred_compute_device).__name__.removeprefix("ML").removesuffix("ComputeDevice")] += 1
        return dict(counts)


def encoder_for(root, models, device, runtime, precision):
    if runtime == "PyTorch":
        checkpoint = models / ("six-references-fp16.pth" if precision == "FP16" else "six-references.pth")
        return TorchEncoder(root, checkpoint, "cpu" if device == "CPU" else "mps", torch.float16 if precision == "FP16" else torch.float32)
    import coremltools as ct
    units = {"CPU": ct.ComputeUnit.CPU_ONLY, "GPU": ct.ComputeUnit.CPU_AND_GPU, "Neural Engine": ct.ComputeUnit.CPU_AND_NE}[device]
    if precision == "INT8":
        return CoreMLEncoder(models / "six-references-int8.mlpackage", units)
    return CoreMLEncoder(fp16_package(root, models), units)


def fp16_package(root, models):
    destination = root / "work" / "six-references-fp16.mlpackage"
    if not destination.exists():
        import coremltools as ct
        import quantize
        model, _ = quantize.load_float_model(root, models / "six-references.pth")
        with torch.no_grad():
            traced = torch.jit.trace(model, torch.zeros(BATCH, 3, 96, 48), check_trace=False)
        ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   inputs=[ct.TensorType(name="crops", shape=(BATCH, 3, 96, 48), dtype=np.float32)],
                   outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
                   compute_precision=ct.precision.FLOAT16).save(str(destination))
    return destination


def test_sessions(root):
    _, splits, _ = train.load_manifest(root)
    rng = random.Random(six_references.EVALUATION_SEED)
    sessions = []
    for entry, clip, bank, episode in six_references.sessions_for(splits["test"], train.ClipCache(12), root / "work" / "references"):
        support, query = six_references.make_session(clip, bank, episode, rng, True)
        sessions.append({"game": entry["game"], "clip": entry["clip"], "support": np.stack(support), "crops": clip["x"][query], "truth": clip["y"][query]})
    return sessions


def classify(encoder, support, crops):
    prototypes = unit(unit(encoder(support)).reshape(4, six_references.REFERENCES, -1).mean(axis=1))
    return (unit(encoder(crops)) @ prototypes.T).argmax(axis=1)


def median_ms(function, repeats=40):
    for _ in range(10):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return 1000 * statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/formats.json"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Torch version")
    torch.set_num_threads(8)
    root = args.root.resolve()
    sessions = test_sessions(root)
    timing_support, timing_crops = sessions[0]["support"], sessions[0]["crops"][:BATCH]
    rows = []
    for device, runtime, precision in FORMATS:
        encoder = encoder_for(root, args.models, device, runtime, precision)
        records = [{"game": s["game"], "clip": s["clip"], "truth": s["truth"], "pred": classify(encoder, s["support"], s["crops"])} for s in sessions]
        metrics = train.statistics(records)
        prototypes = unit(unit(encoder(timing_support)).reshape(4, six_references.REFERENCES, -1).mean(axis=1))
        row = {
            "device": device, "runtime": runtime, "precision": precision,
            "accuracy": metrics["accuracy"], "class_mean": metrics["macro_class_accuracy"], "class_recall": metrics["class_recall"],
            "boxes": metrics["count"], "correct": int(np.trace(metrics["confusion_matrix"])),
            "batch_20_ms": median_ms(lambda: (unit(encoder(timing_crops)) @ prototypes.T).argmax(axis=1)),
            "references_24_ms": median_ms(lambda: encoder(timing_support)),
            "confusion_matrix": metrics["confusion_matrix"],
        }
        if runtime == "Core ML":
            row["convolution_placement"] = encoder.placement()
        row["frames_per_second"] = 1000 / row["batch_20_ms"]
        rows.append(row)
        print(f"{device:13} {runtime:8} {precision}: accuracy {100 * row['accuracy']:.2f}%  class mean {100 * row['class_mean']:.2f}%  "
              f"20 boxes {row['batch_20_ms']:.2f} ms  {row.get('convolution_placement', '')}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train.atomic_json(args.out, {"machine": "Apple M4 Pro", "torch": torch.__version__, "boxes_per_batch": BATCH, "formats": rows})


if __name__ == "__main__":
    main()
