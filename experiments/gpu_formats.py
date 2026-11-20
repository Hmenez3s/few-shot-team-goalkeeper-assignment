import argparse
import copy
import json
import re
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

BATCH = 20
FORMATS = (
    "torch_fp32", "torch_fp16", "trt_fp16", "trt_int8", "trt_fp8",
    "trt_fp16_sparse", "trt_int8_sparse", "trt_fp16_sparse_dense",
    "trt_int8_sparse_dense", "sim_nvfp4", "sim_int4",
    "sim_nvfp4_sparse", "sim_int4_sparse",
)
MODEL_OPT_EXPORT_COMMIT = "187071d48ddaee2155be3a0fbd34eb265ce7229e"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_rows(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)


def scoring_metrics(truth, predictions):
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predictions.shape or truth.ndim != 1:
        raise ValueError("truth and predictions must be same-length vectors")
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    denominators = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), denominators, out=np.zeros(4, dtype=float), where=denominators != 0)
    return {
        "score": float(recalls.mean()),
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "per_class": recalls.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def batch_crops(crops, device, dtype=torch.float32):
    images = torch.from_numpy(np.asarray(crops).copy()).permute(0, 3, 1, 2).contiguous().float().div_(255)
    return ((images - MEAN) / STD).to(device=device, dtype=dtype)


def make_model(checkpoint, device, dtype=torch.float32):
    model = resnet18(weights=None)
    model.fc = nn.Identity()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def embed(model, crops, device, dtype=torch.float32):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(crops), BATCH):
            chunk = crops[start:start + BATCH]
            count = len(chunk)
            if count < BATCH:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], BATCH - count, axis=0)))
            outputs.append(model(batch_crops(chunk, device, dtype))[:count].float().cpu())
    return torch.cat(outputs).numpy()


def aggregate_sessions(encoder, sessions):
    all_truth, all_predictions = [], []
    for session in sessions:
        support = normalize_rows(torch.from_numpy(encoder(session["support"])))
        prototypes = normalize_rows(support.reshape(4, 6, -1).mean(dim=1))
        queries = normalize_rows(torch.from_numpy(encoder(session["crops"])))
        all_predictions.extend((queries @ prototypes.T).argmax(dim=1).numpy().tolist())
        all_truth.extend(session["truth"].astype(np.int64).tolist())
    return scoring_metrics(all_truth, all_predictions)


def read_bundle(path):
    with np.load(path, allow_pickle=False) as bundle:
        session_meta = json.loads(str(bundle["sessions"].item()))["sessions"]
        sessions = [{"support": bundle[f"support_{i}"], "crops": bundle[f"crops_{i}"],
                     "truth": bundle[f"truth_{i}"]} for i in range(len(session_meta))]
        return sessions, bundle["calibration"].copy(), bundle["mean"].copy(), bundle["std"].copy()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_ms(run, device):
    for _ in range(10):
        run()
    synchronize(device)
    samples = []
    for _ in range(40):
        synchronize(device)
        start = time.perf_counter()
        run()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def load_trt():
    import tensorrt as trt
    return trt


def export_onnx(model, onnx_path, device):
    dummy = torch.zeros(BATCH, 3, 96, 48, device=device)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["crops"], output_names=["embedding"],
                      opset_version=20, do_constant_folding=True, dynamo=False)


def build_engine(onnx_path, engine_path, precision, sparse=False):
    trt = load_trt()
    class CapturingLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.lines = []

        def log(self, severity, message):
            self.lines.append(str(message))

    logger = CapturingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision in ("int8", "fp8"):
        if hasattr(trt.BuilderFlag, "STRONGLY_TYPED"):
            config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)
    if sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")
    engine_path.write_bytes(bytes(serialized))
    sparse_lines = [line for line in logger.lines if "sparse" in line.lower()]
    eligible_matches = [re.search(r"(?:(\d+)\s+layers? eligible|eligible for sparse math\D*(\d+))", line, re.IGNORECASE) for line in sparse_lines]
    eligible_counts = [int(match.group(1) or match.group(2)) for match in eligible_matches if match]
    chosen = sum("sparse tactic" in line.lower() for line in sparse_lines)
    return engine_path.stat().st_size, sparse_lines, {
        "eligible_layers": max(eligible_counts) if eligible_counts else None,
        "eligible_log_lines": sum("eligible for sparse math" in line.lower() for line in sparse_lines),
        "chosen_layers": chosen,
    }


class TrtEncoder:
    def __init__(self, engine_path, device):
        trt = load_trt()
        self.device = device
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT failed to deserialize engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream(device)
        tensors = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in tensors if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in tensors if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        self.input_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_name)))).dtype
        self.output_dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)))).dtype

    def __call__(self, crops):
        result = []
        with torch.inference_mode():
            for start in range(0, len(crops), BATCH):
                chunk = crops[start:start + BATCH]
                count = len(chunk)
                if count < BATCH:
                    chunk = np.concatenate((chunk, np.repeat(chunk[-1:], BATCH - count, axis=0)))
                tensor = batch_crops(chunk, self.device, self.input_dtype).contiguous()
                output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
                if output_shape[0] < 0:
                    output_shape = (BATCH, *output_shape[1:])
                output = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
                self.context.set_tensor_address(self.input_name, tensor.data_ptr())
                self.context.set_tensor_address(self.output_name, output.data_ptr())
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
                result.append(output[:count].float().cpu())
        return torch.cat(result).numpy()


def conv_2to4_violations(model):
    violations = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels >= 4 and module.in_channels % 4 == 0:
            weight = module.weight.detach().reshape(module.out_channels, module.in_channels // 4, 4, *module.kernel_size)
            zeros = (weight == 0).sum(dim=2)
            bad = int((zeros < 2).sum().item())
            if bad:
                violations.append({"layer": name, "bad_groups": bad})
    return violations


def modelopt_quantize(model, config, calibration, device):
    import modelopt.torch.quantization as mtq

    def forward_loop(quantized_model):
        for start in range(0, len(calibration), BATCH):
            quantized_model(batch_crops(calibration[start:start + BATCH], device))

    prepared = mtq.quantize(model, copy.deepcopy(config), forward_loop=forward_loop)
    return prepared, mtq


def enabled_conv_quantizers(model, mtq):
    quantizer_type = getattr(getattr(mtq, "nn", None), "TensorQuantizer", None)
    if quantizer_type is None:
        from modelopt.torch.quantization.nn import TensorQuantizer as quantizer_type
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count += sum(isinstance(child, quantizer_type) and child.is_enabled
                         for child in module.modules() if child is not module)
    return count


def modelopt_config(mtq, fmt):
    name = "INT4_BLOCKWISE_WEIGHT_ONLY_CFG" if fmt == "int4" else "NVFP4_DEFAULT_CFG"
    config = getattr(mtq, name, None)
    if config is None:
        raise RuntimeError(f"ModelOpt does not expose {name}")
    return copy.deepcopy(config)


def quantized_format(model, kind, calibration, device):
    import modelopt.torch.quantization as mtq
    config = modelopt_config(mtq, kind)
    if kind == "int8":
        config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    elif kind == "fp8":
        config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    quantized, mtq = modelopt_quantize(model, config, calibration, device)
    return quantized, mtq


def run_simulation(name, checkpoint, sessions, calibration, device):
    kind = "nvfp4" if "nvfp4" in name else "int4"
    model = make_model(checkpoint, device)
    model, mtq = quantized_format(model, kind, calibration, device)
    conv_count = enabled_conv_quantizers(model, mtq)
    if conv_count == 0:
        raise RuntimeError(f"{kind} configuration enabled no Conv2d quantizers")
    result = aggregate_sessions(lambda crops: embed(model, crops, device), sessions)
    return {**result, "simulated": True, "notes": [f"fake quantization only; enabled Conv2d quantizers: {conv_count}"],
            "modelopt_config": kind.upper(), "checkpoint_bytes": checkpoint.stat().st_size, "sparse_log": []}


def main():
    parser = argparse.ArgumentParser(description="Accuracy and latency benchmark for the six-reference ResNet-18")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/six-references.pth"))
    parser.add_argument("--sparse-checkpoint", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workdir", type=Path, default=Path("gpu-format-work"))
    parser.add_argument("--out", type=Path, default=Path("gpu-formats.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    sessions, calibration, mean, std = read_bundle(args.bundle)
    MEAN.copy_(torch.from_numpy(mean).reshape(1, 3, 1, 1))
    STD.copy_(torch.from_numpy(std).reshape(1, 3, 1, 1))
    args.workdir.mkdir(parents=True, exist_ok=True)
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    invalid = set(formats) - set(FORMATS)
    if invalid:
        parser.error(f"unknown formats: {sorted(invalid)}")
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": None, "compute_capability": None,
                   "driver": None, "tensorrt": None, "modelopt": None, "onnx": None}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        environment["gpu"] = props.name
        environment["compute_capability"] = list(torch.cuda.get_device_capability(device))
        try:
            environment["driver"] = torch._C._cuda_getDriverVersion()
        except (AttributeError, RuntimeError):
            environment["driver"] = "unavailable"
    for module_name, key in (("tensorrt", "tensorrt"), ("modelopt", "modelopt"), ("onnx", "onnx")):
        try:
            module = __import__(module_name)
            environment[key] = getattr(module, "__version__", "installed")
        except Exception:
            pass
    expected_matrix = None
    if args.expected:
        expected_matrix = json.loads(args.expected.read_text())["test"]["confusion_matrix"]
    rows = {}
    for name in formats:
        row = {"status": "failed", "simulated": name.startswith("sim_"), "score": None, "accuracy": None,
               "per_class": None, "confusion_matrix": None, "batch_20_ms": None, "notes": [], "sparse_log": []}
        try:
            sparse = "_sparse" in name
            if name.startswith("torch_"):
                dtype = torch.float16 if name == "torch_fp16" else torch.float32
                model = make_model(args.checkpoint, device, dtype)
                encoder = lambda crops: embed(model, crops, device, dtype)
                metrics = aggregate_sessions(encoder, sessions)
                row.update(metrics)
                timing = batch_crops(sessions[0]["crops"][:BATCH], device, dtype)
                with torch.inference_mode():
                    support_features = normalize_rows(torch.from_numpy(encoder(sessions[0]["support"])))
                    prototypes = normalize_rows(support_features.reshape(4, 6, -1).mean(dim=1)).to(device)
                    row["batch_20_ms"] = latency_ms(lambda: normalize_rows(model(timing).float()) @ prototypes.T, device)
                row["checkpoint_bytes"] = args.checkpoint.stat().st_size
            elif name.startswith("sim_"):
                checkpoint = args.sparse_checkpoint if sparse else args.checkpoint
                if checkpoint is None:
                    raise ValueError("sparse simulation requires --sparse-checkpoint")
                row.update(run_simulation(name, checkpoint, sessions, calibration, device))
            else:
                if device.type != "cuda":
                    raise RuntimeError("TensorRT formats require a CUDA device")
                if name == "trt_fp8" and torch.cuda.get_device_capability(device) < (8, 9):
                    row.update(status="skipped", notes=["FP8 TensorRT requires compute capability 8.9 or newer"])
                    rows[name] = row
                    continue
                sparse_build = name.endswith("_sparse") and not name.endswith("_sparse_dense")
                checkpoint = args.sparse_checkpoint if sparse else args.checkpoint
                if checkpoint is None:
                    raise ValueError("sparse TensorRT format requires --sparse-checkpoint")
                model = make_model(checkpoint, device)
                if sparse:
                    violations = conv_2to4_violations(model)
                    row["sparsity_violations"] = violations
                    if violations:
                        raise ValueError(f"checkpoint violates 2:4 in {len(violations)} convolution layers")
                precision = "fp16" if "fp16" in name else ("int8" if "int8" in name else "fp8")
                if precision in ("int8", "fp8"):
                    quantized, mtq = quantized_format(model, precision, calibration, device)
                    row["modelopt_conv_quantizers"] = enabled_conv_quantizers(quantized, mtq)
                    if not row["modelopt_conv_quantizers"]:
                        raise RuntimeError(f"ModelOpt {precision} config enabled no Conv2d quantizers")
                    model = quantized
                onnx_path = args.workdir / f"{name}.onnx"
                engine_path = args.workdir / f"{name}.plan"
                export_onnx(model, onnx_path, device)
                engine_bytes, sparse_lines, sparse_counts = build_engine(onnx_path, engine_path, precision, sparse_build)
                row["sparse_log"] = sparse_lines
                row["sparse_layer_counts"] = sparse_counts
                encoder = TrtEncoder(engine_path, device)
                row.update(aggregate_sessions(encoder, sessions))
                crops = sessions[0]["crops"][:BATCH]
                support = normalize_rows(torch.from_numpy(encoder(sessions[0]["support"])))
                prototypes = normalize_rows(support.reshape(4, 6, -1).mean(dim=1)).to(device)
                normalized = batch_crops(crops, device, encoder.input_dtype)
                def timed_inference():
                    x = torch.cat([normalized, normalized[-1:].expand(BATCH - len(normalized), -1, -1, -1)]) if len(normalized) < BATCH else normalized
                    y = torch.empty((BATCH, 512), device=device, dtype=encoder.output_dtype)
                    encoder.context.set_tensor_address(encoder.input_name, x.contiguous().data_ptr())
                    encoder.context.set_tensor_address(encoder.output_name, y.data_ptr())
                    encoder.context.execute_async_v3(encoder.stream.cuda_stream)
                    return y.float() @ prototypes.T
                row["batch_20_ms"] = latency_ms(timed_inference, device)
                row["engine_bytes"] = engine_bytes
                row["checkpoint_bytes"] = checkpoint.stat().st_size
                row["notes"].append(f"ModelOpt ONNX export approach commit {MODEL_OPT_EXPORT_COMMIT}") if precision in ("int8", "fp8") else None
            row.setdefault("status", "ok")
            row["status"] = "ok"
            if expected_matrix is not None and name == "torch_fp32":
                row["expected_confusion_match"] = row["confusion_matrix"] == expected_matrix
                if not row["expected_confusion_match"]:
                    row["expected_confusion_diff"] = {"expected": expected_matrix, "actual": row["confusion_matrix"]}
                    print(json.dumps(row["expected_confusion_diff"], indent=2), flush=True)
            print(f"{name}: {row['status']} accuracy={row.get('accuracy')} score={row.get('score')} latency_ms={row.get('batch_20_ms')}", flush=True)
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
            print(f"{name}: failed: {row['error']}", flush=True)
        rows[name] = row
    output = {"environment": environment, "formats": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
