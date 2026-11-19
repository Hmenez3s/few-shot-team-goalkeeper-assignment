import argparse
import statistics
import time

import numpy as np
import torch
from PIL import Image

import prepare
import six_references
import train

FRAME_SIZE = (1920, 1080)


def timed(function, synchronize, repeats):
    for _ in range(5):
        function()
        synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/inputs/resnet18.pth")
    parser.add_argument("--checkpoint", default="models/six-references.pth")
    parser.add_argument("--boxes", type=int, default=20)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    synchronize = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(device.type, lambda: None)
    rng = np.random.default_rng(0)
    model = train.build_model(args.weights, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    per_class = state.get("references_per_class", 1)
    reference_crops = rng.integers(0, 256, (4 * per_class, 96, 48, 3), dtype=np.uint8)
    frame = Image.fromarray(rng.integers(0, 256, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
    boxes = [{"x_center": 80 + 90 * i, "y_center": 540, "w": 40, "h": 100} for i in range(args.boxes)]
    with torch.inference_mode():
        def references():
            return six_references.class_prototypes(model(six_references.batch_crops(reference_crops, device)), per_class)

        prototypes = references()
        crops = np.stack([prepare.crop_box(frame, box) for box in boxes])

        def crop():
            return [prepare.crop_box(frame, box) for box in boxes]

        def classify():
            return train.cosine_logits(model(six_references.batch_crops(crops, device)), prototypes).argmax(dim=1).cpu()

        def one_by_one():
            return [train.cosine_logits(model(six_references.batch_crops(crops[i:i + 1], device)), prototypes).argmax(dim=1).cpu() for i in range(len(crops))]

        setup = timed(references, synchronize, args.repeats)
        cropping = timed(crop, lambda: None, args.repeats)
        batched = timed(classify, synchronize, args.repeats)
        single = timed(one_by_one, synchronize, max(5, args.repeats // 3))
    frame_time = cropping + batched
    print(f"device: {device}")
    print(f"reference setup, {4 * per_class} crops, once per match: {1000 * setup:.2f} ms")
    print(f"crop {args.boxes} boxes from a {FRAME_SIZE[0]}x{FRAME_SIZE[1]} frame: {1000 * cropping:.2f} ms")
    print(f"classify {args.boxes} boxes in one batch: {1000 * batched:.2f} ms")
    print(f"per frame: {1000 * frame_time:.2f} ms, {1 / frame_time:.0f} frames per second")
    print(f"classify {args.boxes} boxes one at a time: {1000 * single:.2f} ms")


if __name__ == "__main__":
    main()
