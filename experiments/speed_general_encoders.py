import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import resnet18

BATCH = 20
WARMUP = 5
REPEATS = 30


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def median_ms(model, batch, device):
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(batch)
        synchronize(device)
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            model(batch)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def flops_per_crop(model, height, width):
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        model(torch.rand(1, 3, height, width))
    return counter.get_total_flops() // 2


class Vision(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images):
        return self.encoder(pixel_values=images).pooler_output


def encoders(checkpoint):
    from transformers import CLIPVisionModel, SiglipVisionModel

    ours = resnet18()
    ours.fc = nn.Identity()
    ours.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    siglip = Vision(SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224"))
    clip = Vision(CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32"))
    return {"reference_classifier": (ours, 96, 48), "siglip": (siglip, 224, 224), "clip": (clip, 224, 224)}


def roboflow_clustering_ms(tracks=24, size=768):
    import umap
    from sklearn.cluster import KMeans

    features = np.random.default_rng(0).standard_normal((tracks, size)).astype(np.float32)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        reduced = umap.UMAP(n_components=3, random_state=0).fit_transform(features)
        KMeans(n_clusters=2, random_state=0).fit_predict(reduced)
        times.append((time.perf_counter() - start) * 1000)
    return {"first_call_ms": times[0], "median_later_calls_ms": statistics.median(times[1:])}


def main():
    parser = argparse.ArgumentParser(description="Compare encoder cost of the reference classifier with frozen SigLIP and CLIP")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    import transformers
    result = {"batch": BATCH, "repeats": REPEATS, "torch": torch.__version__, "transformers": transformers.__version__,
              "machine": platform.machine(), "threads": args.threads, "models": {}}
    for name, (model, height, width) in encoders(args.checkpoint).items():
        model.eval()
        entry = {"input_hw": [height, width], "parameters": sum(p.numel() for p in model.parameters()),
                 "multiply_accumulates_per_crop": flops_per_crop(model, height, width), "batch_ms": {}}
        for device_name in devices:
            device = torch.device(device_name)
            model.to(device)
            entry["batch_ms"][device_name] = median_ms(model, torch.rand(BATCH, 3, height, width, device=device), device)
            model.to("cpu")
        result["models"][name] = entry
        print(name, json.dumps(entry), flush=True)
    result["roboflow_umap_kmeans_per_clip"] = roboflow_clustering_ms()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["roboflow_umap_kmeans_per_clip"]))


if __name__ == "__main__":
    main()
