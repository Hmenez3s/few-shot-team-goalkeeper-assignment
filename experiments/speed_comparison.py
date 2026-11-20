import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision.models import resnet18
from yacs.config import CfgNode as YCN

BATCH = 20
WARMUP = 5
REPEATS = 30


def multiply_accumulates(model, sample):
    total = [0]

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            total[0] += output.numel() * kernel
        elif isinstance(module, nn.Linear):
            total[0] += output.numel() * module.in_features

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(sample)
    for hook in hooks:
        hook.remove()
    return total[0] // sample.shape[0]


def median_ms(run, device):
    def synchronize():
        if device.type == "mps":
            torch.zeros(1, device=device).cpu()
    with torch.no_grad():
        for _ in range(WARMUP):
            run()
        synchronize()
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            run()
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def kmeans_ms(embedding_size):
    from sklearn.cluster import KMeans

    tracks = np.random.default_rng(0).standard_normal((24, embedding_size)).astype(np.float32)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        KMeans(n_clusters=2, random_state=0).fit_predict(tracks)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def reference_classifier(checkpoint):
    model = resnet18()
    model.fc = nn.Identity()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
    return model.eval()


def prtreid_model(prtreid, sng, weights):
    sys.path[:0] = [str(prtreid), str(sng)]
    from prtreid.data.datasets import register_image_dataset
    from prtreid.data.datasets.image import SoccerNet
    from prtreid.scripts.main import build_config
    from prtreid.tools.feature_extractor import FeatureExtractor

    register_image_dataset("SoccerNet", SoccerNet, "sn")
    cfg = OmegaConf.load(sng / "sn_gamestate/configs/modules/reid/prtreid.yaml").cfg
    cfg.data.root = str(weights / "dummy-data")
    cfg.data.workers = 1
    cfg.model.load_weights = str(weights / "prtreid-soccernet-baseline.pth.tar")
    cfg.model.bpbreid.hrnet_pretrained_path = str(weights)
    cfg.model.bpbreid.masks.dir = "pose_on_img_crops"
    cfg.use_gpu = False
    cfg = build_config(config=YCN(OmegaConf.to_container(cfg, resolve=True)))
    extractor = FeatureExtractor(cfg, model_path=cfg.model.load_weights, device="cpu",
                                 image_size=(cfg.data.height, cfg.data.width), verbose=False)
    return extractor.model.eval(), (cfg.data.height, cfg.data.width)


class Forward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images, external_parts_masks=None)


def main():
    parser = argparse.ArgumentParser(description="Compare encoder cost of the reference classifier and the SoccerNet PRTReID baseline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prtreid", type=Path, required=True)
    parser.add_argument("--sng", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    ours = reference_classifier(args.checkpoint)
    theirs, size = prtreid_model(args.prtreid.resolve(), args.sng.resolve(), args.weights.resolve())
    theirs = Forward(theirs).eval()
    models = {"reference_classifier": (ours, (96, 48), ("cpu", "mps")), "prtreid": (theirs, size, ("cpu",))}
    result = {"batch": BATCH, "repeats": REPEATS, "torch": torch.__version__, "machine": platform.machine(),
              "processor": platform.processor(), "threads": args.threads, "models": {},
              "note": "PRTReID is measured on CPU only: with torch 1.13.1 its forward pass aborts on MPS with a Metal buffer assertion"}
    for name, (model, (height, width), devices) in models.items():
        sample = torch.rand(BATCH, 3, height, width)
        entry = {"input_hw": [height, width], "parameters": sum(p.numel() for p in model.parameters()),
                 "macs_per_crop": multiply_accumulates(model, sample[:1]), "batch_ms": {}}
        for device_name in devices:
            if device_name == "mps" and not torch.backends.mps.is_available():
                continue
            device = torch.device(device_name)
            try:
                model.to(device)
                batch = sample.to(device)
                entry["batch_ms"][device_name] = median_ms(lambda: model(batch), device)
            except Exception as error:
                entry["batch_ms"][device_name] = f"failed: {type(error).__name__}: {error}"
            model.to("cpu")
        result["models"][name] = entry
        print(name, json.dumps(entry), flush=True)
    result["kmeans_two_clusters_24_tracks_ms"] = kmeans_ms(256)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
