import argparse
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

SPLITS = ("train", "valid", "test")
SIDES = ("left", "right")


def clip_summary(archive, member):
    data = json.loads(archive.read(member))
    info = data["info"]
    jerseys = {side: set() for side in SIDES}
    keepers = defaultdict(list)
    for annotation in data["annotations"]:
        attributes = annotation.get("attributes") or {}
        side = attributes.get("team")
        if side not in SIDES:
            continue
        if attributes.get("role") == "player" and str(attributes.get("jersey") or "").isdigit():
            jerseys[side].add(int(attributes["jersey"]))
        elif attributes.get("role") == "goalkeeper":
            pitch = annotation.get("bbox_pitch") or {}
            if isinstance(pitch.get("x_bottom_middle"), (int, float)):
                keepers[side].append(pitch["x_bottom_middle"])
    shared = jerseys["left"] & jerseys["right"]
    return {
        "member": member,
        "clip": info["id"],
        "game": info["game_id"],
        "half": str(info["game_time_start"])[0],
        "game_time_start": info["game_time_start"],
        "jerseys": {side: sorted(jerseys[side] - shared) for side in SIDES},
        "goalkeeper_median_pitch_x": {side: round(statistics.median(xs), 1) for side, xs in keepers.items()},
        "data": data,
    }


def agreement(a, b):
    same = sum(len(set(a["jerseys"][s]) & set(b["jerseys"][s])) for s in SIDES)
    crossed = len(set(a["jerseys"]["left"]) & set(b["jerseys"]["right"])) + len(set(a["jerseys"]["right"]) & set(b["jerseys"]["left"]))
    return same, crossed


def crops(archive, clip, role, side, count=6):
    data = clip["data"]
    folder = clip["member"].rsplit("/", 1)[0]
    images = {row["image_id"]: row["file_name"] for row in data["images"]}
    rows = [a for a in data["annotations"]
            if (a.get("attributes") or {}).get("role") == role and (a.get("attributes") or {}).get("team") == side
            and isinstance(a.get("bbox_image"), dict) and a["bbox_image"].get("h", 0) >= 60]
    seen, picked = set(), []
    for annotation in sorted(rows, key=lambda a: -a["bbox_image"]["h"]):
        key = (annotation["attributes"].get("jersey"), annotation.get("track_id"))
        if key in seen:
            continue
        seen.add(key)
        picked.append(annotation)
        if len(picked) == count:
            break
    out = []
    for annotation in picked:
        box = annotation["bbox_image"]
        with archive.open(f"{folder}/{data['info']['im_dir']}/{images[annotation['image_id']]}") as stream, Image.open(stream) as frame:
            crop = frame.convert("RGB").crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
        crop.thumbnail((70, 140))
        out.append((crop, "GK" if role == "goalkeeper" else f"#{annotation['attributes'].get('jersey') or '?'}"))
    return out


def sheet(archive, flagged, reference, path):
    rows = []
    for clip in (reference, flagged):
        for side in SIDES:
            rows.append((f"SNGS-{clip['clip']} {side}", crops(archive, clip, "player", side) + crops(archive, clip, "goalkeeper", side, 1)))
    width, height = 170 + 7 * 78, 40 + len(rows) * 165
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), f"Game {flagged['game']}, half {flagged['half']}: outfield players and goalkeeper per 'team' label "
                      f"(reference clip SNGS-{reference['clip']}, flagged clip SNGS-{flagged['clip']})", fill="black")
    for index, (title, items) in enumerate(rows):
        y = 40 + index * 165
        draw.text((8, y + 60), title, fill="black")
        for column, (crop, label) in enumerate(items):
            x = 170 + column * 78
            image.paste(crop, (x, y))
            draw.text((x, y + 145), label, fill="black")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Find SoccerNet GSR clips whose outfield team labels are swapped relative to the rest of the half")
    parser.add_argument("--archives", type=Path, required=True, help="folder with train.zip, valid.zip and test.zip")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for split in SPLITS:
        with zipfile.ZipFile(args.archives / f"{split}.zip") as archive:
            clips = [clip_summary(archive, m) for m in sorted(archive.namelist()) if m.endswith("/Labels-GameState.json")]
            halves = defaultdict(list)
            for clip in clips:
                halves[(clip["game"], clip["half"])].append(clip)
            for (game, half), members in sorted(halves.items()):
                for clip in members:
                    others = [o for o in members if o is not clip]
                    totals = [agreement(clip, other) for other in others]
                    same = sum(s for s, _ in totals)
                    crossed = sum(c for _, c in totals)
                    if crossed <= same:
                        continue
                    reference = max(others, key=lambda other: agreement(clip, other)[1])
                    entry = {
                        "split": split, "clip": f"SNGS-{clip['clip']}", "game": game, "half": half,
                        "game_time_start": clip["game_time_start"],
                        "jersey_pairs_same_side_as_other_clips": same,
                        "jersey_pairs_opposite_side_to_other_clips": crossed,
                        "per_other_clip": {f"SNGS-{o['clip']}": dict(zip(("same", "opposite"), agreement(clip, o))) for o in others},
                        "outfield_jerseys": clip["jerseys"],
                        "reference_clip": f"SNGS-{reference['clip']}",
                        "reference_outfield_jerseys": reference["jerseys"],
                        "goalkeeper_median_pitch_x": clip["goalkeeper_median_pitch_x"],
                        "reference_goalkeeper_median_pitch_x": reference["goalkeeper_median_pitch_x"],
                        "outfield_player_annotations": sum(1 for a in clip["data"]["annotations"] if (a.get("attributes") or {}).get("role") == "player"
                                                           and (a.get("attributes") or {}).get("team") in SIDES),
                        "evidence_image": f"SNGS-{clip['clip']}.jpg",
                    }
                    sheet(archive, clip, reference, args.out / entry["evidence_image"])
                    report.append(entry)
                    print(f"{split} SNGS-{clip['clip']} game {game} half {half}: same={same} opposite={crossed}", flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report)} clips flagged; report in {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
