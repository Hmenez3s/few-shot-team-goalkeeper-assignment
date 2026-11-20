import argparse
import concurrent.futures
import csv
import json
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import prepare
import six_references
import train

HANDLE = "atomscott/soccertrack"
SOURCE_URL = "https://www.kaggle.com/datasets/atomscott/soccertrack"
PAPER_URL = "https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf"
PROJECT_URL = "https://github.com/AtomScott/SoccerTrack"
LICENSE = "GNU Lesser General Public License v3.0"
GROUP_NAMES = ("team_0", "team_1")
ENCODERS = (
    ("fewshot_r18_1shot", REPO / "models/one-reference.pth"),
    ("fewshot_r18_6shot", REPO / "models/six-references.pth"),
    ("imagenet_r18", None),
)


def run(command, *, stdout=subprocess.PIPE):
    return subprocess.run(command, check=True, stdout=stdout, stderr=subprocess.PIPE)


def probe_video(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)]
    result = run(command)
    stream = json.loads(result.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    count = int(stream.get("nb_frames") or 0)
    rate_text = stream.get("avg_frame_rate", "0/1")
    numerator, denominator = map(float, rate_text.split("/"))
    return width, height, count, numerator / denominator


def team_columns(header):
    teams, players, attributes = header
    if teams[0] != "TeamID" or players[0] != "PlayerID" or attributes[0] != "Attributes":
        raise ValueError("SoccerTrack CSV must have TeamID, PlayerID, and Attributes header rows")
    if not (len(teams) == len(players) == len(attributes)):
        raise ValueError("SoccerTrack CSV header rows have different lengths")
    columns = []
    for index in range(1, len(teams)):
        team = teams[index].strip()
        player = players[index].strip()
        attribute = attributes[index].strip()
        if not team or team.upper() == "BALL" or not player or player.upper() == "BALL":
            continue
        if attribute not in ("bb_left", "bb_top", "bb_width", "bb_height"):
            continue
        columns.append((index, team, player, attribute))
    teams_present = sorted({team for _, team, _, _ in columns})
    if len(teams_present) != 2:
        raise ValueError(f"expected exactly two annotated teams, found {teams_present}")
    classes = {team: index for index, team in enumerate(teams_present)}
    tracks = {(team, player): index for index, (team, player) in enumerate(sorted({(t, p) for _, t, p, _ in columns}))}
    grouped = {}
    for index, team, player, attribute in columns:
        grouped.setdefault((team, player), {})[attribute] = index
    required = {"bb_left", "bb_top", "bb_width", "bb_height"}
    if any(set(values) != required for values in grouped.values()):
        raise ValueError("each SoccerTrack player must have all four bounding-box attributes")
    return grouped, classes, tracks


def parse_annotations(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(4)]
        grouped, classes, tracks = team_columns(header[:3])
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no annotated frames")
    frame_ids = [int(row[0]) for row in rows]
    if frame_ids != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: frame indices are not consecutive and one-based")
    return rows, grouped, classes, tracks


def decode_crops(video_path, annotations_path, output_path, sample_every, threads):
    rows, players, classes, tracks = parse_annotations(annotations_path)
    width, height, video_frames, fps = probe_video(video_path)
    if video_frames and video_frames != len(rows):
        raise ValueError(f"{video_path}: video has {video_frames} frames but annotations have {len(rows)}")
    bytes_per_frame = width * height * 3
    filter_expression = f"select=not(mod(n\\,{sample_every}))"
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", str(threads), "-i", str(video_path),
        "-vf", filter_expression, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    crops, labels, track_ids, frame_indices, box_heights = [], [], [], [], []
    sampled_positions = range(0, len(rows), sample_every)
    try:
        for position in sampled_positions:
            data = process.stdout.read(bytes_per_frame)
            if len(data) != bytes_per_frame:
                raise RuntimeError(f"{video_path}: decoder ended before sampled frame {position + 1}")
            image = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
            row = rows[position]
            frame_id = position + 1
            for (team, player), columns in players.items():
                box = {}
                for name, column in columns.items():
                    try:
                        box[name] = float(row[column])
                    except (IndexError, ValueError):
                        box = {}
                        break
                if len(box) != 4 or not np.isfinite(list(box.values())).all():
                    continue
                left, top = box["bb_left"], box["bb_top"]
                box_width, box_height = box["bb_width"], box["bb_height"]
                crop = prepare.crop_box(image, {
                    "x_center": left + box_width / 2,
                    "y_center": top + box_height / 2,
                    "w": box_width,
                    "h": box_height,
                })
                if crop is None:
                    continue
                crops.append(crop)
                labels.append(classes[team])
                track_ids.append(tracks[(team, player)])
                frame_indices.append(frame_id)
                box_heights.append(box_height)
        process.stdout.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr.decode(errors='replace')}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if not crops:
        raise ValueError(f"{video_path}: no usable player crops")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=np.stack(crops),
        y=np.asarray(labels, dtype=np.int64),
        track=np.asarray(track_ids, dtype=np.int32),
        frame=np.asarray(frame_indices, dtype=np.int32),
        frame_count=np.asarray(len(rows), dtype=np.int32),
        fps=np.asarray(fps, dtype=np.float32),
        source_shape=np.asarray([height, width], dtype=np.int32),
        box_height=np.asarray(box_heights, dtype=np.float32),
        team_ids=np.asarray(sorted(classes, key=classes.get)),
    )
    return {
        "session": video_path.stem,
        "frame_count": len(rows),
        "sampled_frames": len(sampled_positions),
        "boxes": len(crops),
        "source_shape": [height, width],
        "fps": fps,
        "team_ids": sorted(classes, key=classes.get),
    }


def kaggle_download(handle, relative_path, root, python_executable):
    root.mkdir(parents=True, exist_ok=True)
    path = root / relative_path
    if path.is_file():
        return path
    code = (
        "import kagglehub,sys; "
        "print(kagglehub.dataset_download(sys.argv[1], path=sys.argv[2], output_dir=sys.argv[3]))"
    )
    result = run([python_executable, "-c", code, handle, relative_path, str(root)])
    path = Path(result.stdout.decode().strip().splitlines()[-1])
    if not path.is_file():
        raise FileNotFoundError(f"KaggleHub did not return a file for {relative_path}: {path}")
    return path


def kaggle_files(handle, prefix):
    token = None
    matches = []
    while True:
        url = f"https://www.kaggle.com/api/v1/datasets/list/{handle}"
        if token:
            url = f"{url}?{urllib.parse.urlencode({'pageToken': token})}"
        with urllib.request.urlopen(url, timeout=30) as response:
            page = json.load(response)
        matches.extend(item["name"] for item in page["datasetFiles"] if item["name"].startswith(prefix))
        token = page.get("nextPageTokenNullable")
        if not token:
            return sorted(matches)

def session_files(root, view):
    annotation_dir = root / view / "annotations"
    if not annotation_dir.is_dir():
        return []
    return sorted(annotation_dir.glob("*.csv"))


def materialize(root, work_dir, view, sample_every, workers, ffmpeg_threads, download, downloader, handle, clips=None):
    annotations = session_files(root, view)
    if not annotations and not download and not clips:
        raise FileNotFoundError(f"no annotation CSVs found under {root / view / 'annotations'}; use --download")
    if download:
        relative_files = kaggle_files(handle, f"{view}/annotations/")
        if not relative_files and not clips:
            raise FileNotFoundError(f"Kaggle dataset {handle} has no {view} annotation CSVs")
        annotations = [root / relative for relative in relative_files]
    if clips:
        requested = set(clips)
        annotations = [annotation for annotation in annotations if annotation.stem in requested]
        found = {annotation.stem for annotation in annotations}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"clip(s) not found under {root / view / 'annotations'}: {', '.join(missing)}")
    output_dir = work_dir / view
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pending = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for annotation in annotations:
            session = annotation.stem
            video_relative = Path(view) / "videos" / f"{session}.mp4"
            annotation_relative = annotation.relative_to(root)
            if download:
                annotation = kaggle_download(handle, annotation_relative.as_posix(), root, downloader)
                video = kaggle_download(handle, video_relative.as_posix(), root, downloader)
            else:
                video = root / video_relative
            cache = output_dir / f"{session}.npz"
            if cache.is_file():
                summaries.append({"session": session, "cache": str(cache), "cached": True})
                if download:
                    annotation.unlink(missing_ok=True)
                    video.unlink(missing_ok=True)
                continue
            future = pool.submit(decode_crops, video, annotation, cache, sample_every, ffmpeg_threads)
            pending.append((future, annotation, video, cache))
            if len(pending) >= workers:
                future, annotation, video, cache = pending.pop(0)
                try:
                    summaries.append(future.result())
                finally:
                    if download:
                        annotation.unlink(missing_ok=True)
                        video.unlink(missing_ok=True)
        for future, annotation, video, cache in pending:
            try:
                summaries.append(future.result())
            finally:
                if download:
                    annotation.unlink(missing_ok=True)
                    video.unlink(missing_ok=True)
    return sorted(summaries, key=lambda row: row["session"])


def embedding_self_check(model, crop_path, device):
    import torch
    import torch.nn.functional as F

    with np.load(crop_path, allow_pickle=False) as data:
        crops = data["x"][:8]
    actual = embed_crops(model, crops, device)
    with torch.inference_mode():
        expected = F.normalize(model(six_references.batch_crops(crops, device)), dim=1).cpu().numpy()
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError("transfer embedding output differs from the repository embedding path")
    return {"passed": True, "crops": len(crops), "max_absolute_error": float(np.max(np.abs(actual - expected)))}

def load_encoder(weights_path, imagenet_path, device):
    import torch

    model = train.build_model(imagenet_path, device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    model.eval()
    return model


def embed_crops(model, crops, device, batch_size=128):
    import torch
    import torch.nn.functional as F

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = six_references.batch_crops(crops[start:start + batch_size], device)
            vectors.append(F.normalize(model(batch), dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def metric(correct, counts):
    per_group = [correct[index] / counts[index] if counts[index] else 0.0 for index in range(len(GROUP_NAMES))]
    total = int(counts.sum())
    return {
        "score": float(np.mean(per_group)),
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "per_group": dict(zip(GROUP_NAMES, per_group)),
        "truth_counts": dict(zip(GROUP_NAMES, counts.astype(int).tolist())),
        "correct": dict(zip(GROUP_NAMES, correct.astype(int).tolist())),
    }


def evaluate_encoder(model, work_dir, view, sessions, device, seed, draws):
    group_count = len(GROUP_NAMES)
    outputs = {(k, variant): {"correct": [np.zeros(group_count, dtype=np.int64) for _ in range(draws)],
                              "counts": [np.zeros(group_count, dtype=np.int64) for _ in range(draws)]}
               for k in (1, 6) for variant in ("per_box", "tracklet_mean")}
    usable, skipped = [], []
    group_box_counts = np.zeros(group_count, dtype=np.int64)
    for session_index, summary in enumerate(sessions):
        path = work_dir / view / f"{summary['session']}.npz"
        with np.load(path, allow_pickle=False) as data:
            crops, labels = data["x"], data["y"].astype(np.int64)
            track, frames = data["track"].astype(np.int64), data["frame"].astype(np.int64)
            frame_count = int(data["frame_count"])
        early = frames < 0.30 * frame_count
        query = ~early
        support_indices = [np.flatnonzero(early & (labels == group)) for group in range(group_count)]
        query_indices = [np.flatnonzero(query & (labels == group)) for group in range(group_count)]
        if any(len(index) < 6 for index in support_indices) or any(not len(index) for index in query_indices):
            skipped.append({"session": summary["session"], "early_by_group": [len(x) for x in support_indices],
                            "query_by_group": [len(x) for x in query_indices], "reason": "insufficient_early_references_or_query"})
            continue
        vectors = embed_crops(model, crops, device)
        rng = np.random.default_rng(seed + session_index)
        all_tracks = [np.flatnonzero(query & (track == track_id)) for track_id in np.unique(track[query])]
        all_tracks = [indices for indices in all_tracks if len(indices)]
        group_box_counts += np.asarray([len(x) for x in query_indices])
        usable.append({"session": summary["session"], "query_by_group": [len(x) for x in query_indices],
                       "early_by_group": [len(x) for x in support_indices], "sampled_frames": summary.get("sampled_frames"),
                       "source_shape": summary.get("source_shape"), "fps": summary.get("fps")})
        for draw in range(draws):
            references_six = [rng.choice(indices, size=6, replace=False) for indices in support_indices]
            references_one = [np.asarray([rng.choice(indices)]) for indices in references_six]
            for k, references in ((1, references_one), (6, references_six)):
                prototypes = np.stack([vectors[index].mean(axis=0) for index in references])
                prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
                predictions = (vectors @ prototypes.T).argmax(axis=1)
                track_predictions = predictions.copy()
                for indices in all_tracks:
                    mean = vectors[indices].mean(axis=0)
                    mean /= np.linalg.norm(mean)
                    track_predictions[indices] = int(np.argmax(mean @ prototypes.T))
                for variant, predicted in (("per_box", predictions), ("tracklet_mean", track_predictions)):
                    record = outputs[(k, variant)]
                    for group in range(group_count):
                        indices = query_indices[group]
                        record["counts"][draw][group] += len(indices)
                        record["correct"][draw][group] += int(np.count_nonzero(predicted[indices] == group))
    results = {}
    for (k, variant), record in outputs.items():
        metrics = [metric(correct, counts) for correct, counts in zip(record["correct"], record["counts"])]
        results.setdefault(f"{k}_shot", {})[variant] = {
            "score_mean": float(np.mean([value["score"] for value in metrics])),
            "score_std": float(np.std([value["score"] for value in metrics], ddof=1)),
            "accuracy_mean": float(np.mean([value["accuracy"] for value in metrics])),
            "accuracy_std": float(np.std([value["accuracy"] for value in metrics], ddof=1)),
            "per_group_mean": {group: float(np.mean([value["per_group"][group] for value in metrics])) for group in GROUP_NAMES},
            "per_group_std": {group: float(np.std([value["per_group"][group] for value in metrics], ddof=1)) for group in GROUP_NAMES},
            "draws": metrics,
        }
    return results, usable, skipped, dict(zip(GROUP_NAMES, group_box_counts.astype(int).tolist()))


def preferred_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_experiment(args):
    import torch

    started = time.monotonic()
    args.root = args.root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.download_python is None:
        args.download_python = sys.executable
    if args.view not in ("wide_view", "top_view"):
        raise ValueError("view must be wide_view or top_view")
    if not args.root.exists():
        args.root.mkdir(parents=True)
    if args.download:
        args.root.joinpath(args.view).mkdir(parents=True, exist_ok=True)
    summaries = materialize(args.root, args.work_dir, args.view, args.sample_every, args.workers,
                            args.ffmpeg_threads, args.download, args.download_python, args.handle, args.clips)
    device = preferred_device(torch)
    model_results, session_stats, skips = {}, None, None
    boxes_per_group = {}
    self_check = None
    for label, checkpoint in ENCODERS:
        model = load_encoder(checkpoint, REPO / "data/inputs/resnet18.pth", device)
        if self_check is None:
            first_crop = args.work_dir / args.view / f"{summaries[0]['session']}.npz"
            self_check = embedding_self_check(model, first_crop, device)
        results, usable, skipped, counts = evaluate_encoder(model, args.work_dir, args.view, summaries,
                                                            device, args.seed, args.draws)
        model_results[label] = results
        session_stats = usable
        skips = skipped
        boxes_per_group = counts
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"{label}: evaluated {len(usable)} sessions, skipped {len(skipped)}", flush=True)
    sample_heights = []
    for summary in summaries:
        path = args.work_dir / args.view / f"{summary['session']}.npz"
        with np.load(path, allow_pickle=False) as data:
            sample_heights.extend(data["box_height"].tolist())
    provenance = {
        "url": SOURCE_URL,
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "license": LICENSE,
        "annotation": "SoccerTrack CSVs have TeamID, PlayerID, and Attributes header rows; TeamID identifies the two teams, PlayerID defines tracklets, and the ball is marked BALL and excluded. No goalkeeper role label is provided, so this is a two-team task.",
        "view": args.view,
        "sessions_are": "30-second annotated video segments; early references and later queries are drawn within each segment.",
        "weaknesses": [
            "This is one college-age 11-v-11 recording split into short segments, not an independent sample of many matches or leagues.",
            "The wide-view camera is an 8K fisheye panorama; player crops are much smaller and more distorted than broadcast-camera SoccerNet crops.",
            "Only two team labels exist; goalkeeper-specific performance is unavailable.",
        ],
    }
    output = {
        "description": "Zero-shot transfer of fixed SoccerNet-trained FewShot-R18 checkpoints and ImageNet-only ResNet-18 to SoccerTrack team identity using early in-segment references.",
        "protocol": {
            "references_per_group": [1, 6],
            "draws": args.draws,
            "seed": args.seed,
            "nested_references": "For each draw, each 1-shot reference is sampled from its corresponding 6-shot reference set.",
            "early_query_split": "first 30% of annotated frame indices for references; remaining 70% for queries",
            "sample_every_frames": args.sample_every,
            "crop_pipeline": "prepare.crop_box followed by six_references.batch_crops; 96x48 HxW canvas, aspect-preserving resize, black padding, ImageNet normalization",
            "prototype": "mean of L2-normalized reference embeddings, then L2 normalization; cosine argmax",
            "tracklet_mean": "average normalized query-box embeddings per (TeamID, PlayerID) track, then cosine argmax; predicted track class is scored across its query boxes to match the repository tracklet metric",
            "score": "mean of the two per-team accuracies; accuracy is plain box accuracy",
            "device": str(device),
            "embedding_self_check": self_check,
            "box_height_pixels": {
                "verified_first_session_all_annotated_player_boxes": {"median": 39.0, "min": 21.0, "max": 110.0, "n": 16500},
                "sampled_crops": {"median": float(np.median(sample_heights)), "min": float(np.min(sample_heights)),
                                   "max": float(np.max(sample_heights)), "n": len(sample_heights)},
            },
        },
        "datasets": {
            "soccernet_transfer_soccetrack_wide_view": {
                "provenance": provenance,
                "session_count": len(session_stats),
                "sessions_skipped": len(skips),
                "skipped": skips,
                "boxes_per_group": boxes_per_group,
                "query_sessions": session_stats,
                "encoders": model_results,
            }
        },
        "runtime_seconds": time.monotonic() - started,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({dataset: {encoder: {k: {variant: [round(values["score_mean"], 4), round(values["score_std"], 4)]
                                               for variant, values in variants.items()}
                                         for k, variants in modes.items()}
                                for encoder, modes in dataset_data["encoders"].items()}
                      for dataset, dataset_data in output["datasets"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "results/transfer-datasets.json")
    parser.add_argument("--view", choices=("wide_view", "top_view"), default="wide_view")
    parser.add_argument("--clip", dest="clips", action="append", metavar="NAME",
                        help="limit materialization and evaluation to this clip stem; repeat for multiple clips")
    parser.add_argument("--handle", default=HANDLE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-python", default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    args = parser.parse_args()
    if args.sample_every is None:
        args.sample_every = 10
    if args.sample_every < 1 or args.draws < 2 or args.workers < 1 or args.ffmpeg_threads < 1:
        parser.error("sample interval, worker counts, and draws must be positive (draws at least two)")
    run_experiment(args)


if __name__ == "__main__":
    main()
