import argparse
import collections
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import prepare
import reference_bank
import train

LABELS = ["Player A", "Player B", "Goalkeeper A", "Goalkeeper B"]
SHORT = ["P-A", "P-B", "G-A", "G-B"]
COLORS = ["#ffb763", "#7bdbff", "#ffb763", "#7bdbff"]
HALVES = {"1": "first half", "2": "second half"}
SHEET_UNITS = ("game-7-half-2", "game-11-half-2")
VIDEO_UNIT = "game-7-half-2"
VIDEO_SOURCE_CLIP = "SNGS-129"
VIDEO_FRAMES = 150
VIDEO_STRIDE = 2
VIDEO_SIZE = (1280, 824)
VIDEO_HEADER = 104
PREVIEW_FRAMES = (20, 75, 130)
FONT_DIRS = (Path("/System/Library/Fonts/Supplemental"), Path("/usr/share/fonts/truetype/dejavu"))
FONT_FILES = {False: ("Arial.ttf", "DejaVuSans.ttf"), True: ("Arial Bold.ttf", "DejaVuSans-Bold.ttf")}


def font(size, bold=False):
    for directory in FONT_DIRS:
        for name in FONT_FILES[bold]:
            path = directory / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def load_model(root, device):
    model = train.build_model(root / "inputs" / "resnet18.pth", device)
    locked = torch.load(root / "work" / "checkpoint.pth", map_location=device, weights_only=True)
    model.load_state_dict(locked["model"])
    model.eval()
    return model


def load_session(root, unit):
    _, splits, _ = train.load_manifest(root)
    entries = [entry for entry in splits["test"] if entry["clip"] == unit]
    if not entries:
        raise ValueError(f"{unit} is not a test session")
    clip = train.ClipCache(1).get(entries[0]["path"])
    episode = train.fixed_episode(clip)
    if episode is None:
        raise ValueError(f"{unit} has no four-reference episode")
    return entries[0], clip, episode


def session_title(entry):
    half = entry["clip"].rsplit("-", 1)[1]
    return f"SoccerNet GSR · game {entry['game']}, {HALVES[half]}"


def confusion(truth, predicted):
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def example_sheet(root, model, device, unit, output):
    entry, clip, episode = load_session(root, unit)
    support, query = episode
    query = np.asarray(query)
    truth = clip["y"][query]
    predicted = train.predictions(model, clip, episode, device)
    matrix = confusion(truth, predicted)
    recorded = json.loads((root / "work" / "result.json").read_text())["test"]["trained"]["per_clip"]
    if matrix.tolist() != recorded[f"{entry['game']}/{unit}"]["confusion_matrix"]:
        raise ValueError(f"{unit}: predictions differ from the recorded test result")
    sheet = Image.new("RGB", (1160, 970), "#0d1524")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 24), session_title(entry), font=font(36, True), fill="#f3f6fa")
    draw.text((31, 74), f"Match never seen in training · {matrix.trace()}/{matrix.sum()} boxes correct in this session", font=font(22), fill="#aebfd4")

    def cell(column, top, index, caption, accent):
        left = 30 + column * 278
        draw.rounded_rectangle((left, top, left + 264, top + 229), radius=14, fill="#162237", outline="#273953", width=2)
        if index is not None:
            crop = Image.fromarray(clip["x"][index]).resize((80, 160), Image.Resampling.BICUBIC)
            sheet.paste(crop, (left + 92, top + 12))
        draw.text((left + 132, top + 179), LABELS[column], font=font(21, True), fill="#f3f6fa", anchor="mt")
        draw.text((left + 132, top + 206), caption, font=font(17), fill=accent, anchor="mt")

    draw.text((30, 120), "1  References given to the model", font=font(22, True), fill="#f3f6fa")
    draw.text((30, 394), "2  Later boxes classified correctly", font=font(22, True), fill="#f3f6fa")
    draw.text((30, 668), "3  Later boxes the model got wrong", font=font(22, True), fill="#f3f6fa")
    for class_id in range(4):
        cell(class_id, 151, support[class_id], "Reference", "#9dc6ff")
        correct = np.flatnonzero((truth == class_id) & (predicted == class_id))
        if len(correct):
            cell(class_id, 425, query[correct[len(correct) // 2]], f"Predicted: {LABELS[class_id]}", "#7cf2bc")
        else:
            cell(class_id, 425, None, "No correct box", "#aebfd4")
        errors = matrix[class_id].copy()
        errors[class_id] = 0
        if errors.max():
            wrong = int(errors.argmax())
            missed = np.flatnonzero((truth == class_id) & (predicted == wrong))
            cell(class_id, 699, query[missed[len(missed) // 2]], f"Predicted: {LABELS[wrong]}", "#ff9f99")
        else:
            cell(class_id, 699, None, "No errors", "#aebfd4")
    draw.text((31, 939), "White title = dataset label · A/B are defined per match · crops enlarged", font=font(17), fill="#aebfd4")
    sheet.save(output, optimize=True)
    print(f"{output}: {matrix.trace()}/{matrix.sum()} correct", flush=True)


def keeper_coverage(rows, boxes, start):
    counts = collections.Counter()
    for row in rows[start:start + VIDEO_FRAMES * VIDEO_STRIDE:VIDEO_STRIDE]:
        counts.update(class_id for class_id, _ in boxes[str(row["image_id"])])
    return min(counts[2], counts[3]), counts[2] + counts[3], sum(counts.values())


def video(root, model, device, unit, source_clip, output):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render the video")
    entry, clip, episode = load_session(root, unit)
    support, _ = episode
    if list(clip["y"][support]) != [0, 1, 2, 3]:
        raise ValueError("references must be ordered by class")
    with torch.no_grad():
        prototypes = model(train.batch_images(clip, support, device))
    width, height = VIDEO_SIZE
    large, small, badge = font(26, True), font(15), font(13, True)
    reference_images = [Image.fromarray(clip["x"][index]).resize((26, 52), Image.Resampling.BICUBIC) for index in support]
    matrix = np.zeros((4, 4), dtype=int)
    previews = []
    pending = output.with_name(output.stem + ".pending.mp4")
    with zipfile.ZipFile(root / "archives" / "test.zip") as archive:
        labels = json.loads(archive.read(f"{source_clip}/Labels-GameState.json"))
        info = labels["info"]
        identity = prepare.identity(labels)
        timing = prepare.unit_identity(labels)
        half = str(unit.rsplit("-", 1)[1])
        if (identity is None
                or identity[0] != str(entry["game"])
                or f"SNGS-{identity[1].zfill(3)}" != source_clip
                or int(info["frame_rate"]) != 25
                or timing is None
                or timing[0] != half):
            raise ValueError(f"{source_clip} does not belong to {unit}")
        manifest = json.loads((root / "work" / "manifest.json").read_text(encoding="utf-8"))
        swap_teams = reference_bank.team_swap_for_clip(manifest, "test", entry["game"], half, identity[1])
        boxes = collections.defaultdict(list)
        for annotation in labels["annotations"]:
            class_id = prepare.label_class(annotation, swap_teams)
            if annotation.get("supercategory") == "object" and class_id is not None and isinstance(annotation.get("bbox_image"), dict):
                boxes[str(annotation["image_id"])].append((class_id, annotation["bbox_image"]))
        rows = sorted((row for row in labels["images"] if row.get("is_labeled") is True), key=lambda row: row["file_name"])
        span = VIDEO_FRAMES * VIDEO_STRIDE
        start = max(range(0, len(rows) - span + 1, 25), key=lambda candidate: keeper_coverage(rows, boxes, candidate))
        frames = rows[start:start + span:VIDEO_STRIDE]
        first_time = int(info["clip_start"]) + (int(Path(frames[0]["file_name"]).stem) - 1) * 40
        if first_time <= int(clip["t"][support].max()):
            raise ValueError("video frames must come after every reference")
        encoder = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"25/{VIDEO_STRIDE}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(pending)],
            stdin=subprocess.PIPE,
        )
        try:
            for number, row in enumerate(frames):
                with archive.open(f"{source_clip}/{info['im_dir']}/{row['file_name']}") as stream, Image.open(stream) as source:
                    frame = source.convert("RGB")
                visible = []
                for class_id, bbox in boxes[str(row["image_id"])]:
                    crop = prepare.crop_box(frame, bbox)
                    if crop is not None:
                        visible.append((class_id, bbox, crop))
                canvas = Image.new("RGB", (width, height), "#111b2b")
                canvas.paste(frame.resize((width, height - VIDEO_HEADER), Image.Resampling.BILINEAR), (0, VIDEO_HEADER))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 12), "Four-reference classification", font=large, fill="#ffffff")
                draw.text((17, 49), session_title(entry), font=small, fill="#c9d3e2")
                draw.text((17, 73), "Boxes from the dataset · labels predicted by the model", font=small, fill="#c9d3e2")
                for class_id, reference in enumerate(reference_images):
                    left = 563 + class_id * 171
                    canvas.paste(reference, (left, 12))
                    draw.text((left + 34, 22), LABELS[class_id], font=small, fill=COLORS[class_id])
                    draw.text((left + 34, 44), "Reference", font=small, fill="#cad3df")
                draw.text((562, 77), "P = player  ·  G = goalkeeper  ·  ! = differs from the dataset label", font=small, fill="#cad3df")
                if visible:
                    batch = {"x": np.stack([crop for _, _, crop in visible])}
                    with torch.no_grad():
                        features = model(train.batch_images(batch, list(range(len(visible))), device))
                    predicted = train.cosine_logits(features, prototypes).argmax(dim=1).cpu().tolist()
                    scale_x = width / frame.width
                    scale_y = (height - VIDEO_HEADER) / frame.height
                    for (class_id, bbox, _), guess in zip(visible, predicted):
                        matrix[class_id, guess] += 1
                        cx, cy, w, h = (float(bbox[key]) for key in ("x_center", "y_center", "w", "h"))
                        x0 = max(0, int((cx - w / 2) * scale_x))
                        x1 = min(width - 1, int((cx + w / 2) * scale_x))
                        y0 = max(VIDEO_HEADER, int((cy - h / 2) * scale_y) + VIDEO_HEADER)
                        y1 = min(height - 1, int((cy + h / 2) * scale_y) + VIDEO_HEADER)
                        color = COLORS[guess]
                        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
                        text = SHORT[guess] + (" !" if guess != class_id else "")
                        text_width = int(draw.textlength(text, font=badge)) + 8
                        text_left = min(width - text_width, x0)
                        text_top = max(VIDEO_HEADER, y0 - 16)
                        draw.rounded_rectangle((text_left, text_top, text_left + text_width, text_top + 16), radius=3, fill="#101927", outline=color, width=1)
                        draw.text((text_left + 4, text_top + 1), text, font=badge, fill="#ffaaa5" if guess != class_id else color)
                encoder.stdin.write(canvas.tobytes())
                if number in PREVIEW_FRAMES:
                    previews.append(canvas.resize((960, 618), Image.Resampling.LANCZOS))
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise RuntimeError("ffmpeg failed")
            pending.replace(output)
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            if pending.exists():
                pending.unlink()
    strip = Image.new("RGB", (960, 618 * len(previews)), "#111b2b")
    for position, preview in enumerate(previews):
        strip.paste(preview, (0, position * 618))
    strip.save(output.with_name(output.stem + "-frames.jpg"), quality=90, optimize=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(output), "-vf",
         "fps=6,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
         "-y", str(output.with_suffix(".gif"))],
        check=True, stdin=subprocess.DEVNULL,
    )
    stats = {
        "session": unit, "source_clip": source_clip, "first_frame": frames[0]["file_name"], "last_frame": frames[-1]["file_name"],
        "frames": len(frames), "boxes": int(matrix.sum()), "accuracy": float(matrix.trace() / matrix.sum()),
        "class_names": train.CLASSES, "confusion_matrix": matrix.tolist(),
    }
    train.atomic_json(output.with_name(output.stem + "-stats.json"), stats)
    print(f"{output}: {stats['frames']} frames, {matrix.trace()}/{matrix.sum()} boxes match the dataset label", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(root, device)
    for unit in SHEET_UNITS:
        example_sheet(root, model, device, unit, args.out / f"examples-{unit}.png")
    video(root, model, device, VIDEO_UNIT, VIDEO_SOURCE_CLIP, args.out / "classification-game-7.mp4")


if __name__ == "__main__":
    main()
