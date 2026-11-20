import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from yacs.config import CfgNode as CN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prtreid", type=Path, required=True)
    parser.add_argument("--sng", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        raise ValueError("PRTReID inference is CPU-only")
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import sys
    sys.path[:0] = [str(args.prtreid), str(args.sng)]
    from prtreid.scripts.main import build_config
    from prtreid.tools.feature_extractor import FeatureExtractor
    from prtreid.utils.tools import extract_test_embeddings
    from yacs.config import CfgNode as YCN
    from prtreid.data.datasets import register_image_dataset
    from prtreid.data.datasets.image import SoccerNet
    register_image_dataset("SoccerNet", SoccerNet, "sn")

    weights = args.weights.resolve()
    model_weights = weights / "prtreid-soccernet-baseline.pth.tar"
    hrnet_weights = weights / "hrnetv2_w32_imagenet_pretrained.pth"
    if not model_weights.is_file() or not hrnet_weights.is_file():
        raise FileNotFoundError("official model weights are missing")
    sng = args.sng.resolve()
    cfg = OmegaConf.load(sng / "sn_gamestate/configs/modules/reid/prtreid.yaml")
    cfg = cfg.cfg
    cfg.data.root = str(args.out.parent / "dummy-data")
    cfg.data.workers = 8
    cfg.model.load_weights = str(model_weights)
    cfg.model.bpbreid.hrnet_pretrained_path = str(weights)
    cfg.model.bpbreid.masks.dir = "pose_on_img_crops"
    cfg.use_gpu = False
    cfg = YCN(OmegaConf.to_container(cfg, resolve=True))
    cfg = build_config(config=cfg)
    extractor = FeatureExtractor(cfg, model_path=str(model_weights), device="cpu", image_size=(cfg.data.height, cfg.data.width), verbose=False)
    data = np.load(args.extract, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    pix, offsets, widths, heights = data["pixels"], data["offsets"], data["widths"], data["heights"]
    embeddings, roles = [], []
    for start in range(0, len(meta), args.batch_size):
        crops = []
        for i in range(start, min(start + args.batch_size, len(meta))):
            crops.append(pix[offsets[i]:offsets[i + 1]].reshape((heights[i], widths[i], 3)))
        output = extractor(crops)
        emb, _, _, _, scores = extract_test_embeddings(output, cfg.model.bpbreid.test_embeddings)
        embeddings.extend(emb.detach().cpu().numpy())
        roles.extend(scores["globl"].detach().cpu().numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, embeddings=np.asarray(embeddings), role_logits=np.asarray(roles))
    print(f"wrote {len(embeddings)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
