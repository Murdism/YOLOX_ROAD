import argparse
import json
import os
import shutil
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ROAD Waymo annotations into the YOLOX MOT-style COCO format."
    )
    parser.add_argument(
        "--road-dir",
        dest="road_dir",
        default="datasets/road_uk/train",
        help="ROAD UK dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset root. Defaults to <road-dir>/coco_annotations.",
    )
    parser.add_argument(
        "--annotations-path",
        "--annotations-json",
        dest="annotations_path",
        default="annotations/road_trainval_v1.0.json",
        help=(
            "Annotations JSON path (absolute or relative to --road-dir). "
            "Default: annotations/road_trainval_v1.0.json"
        ),
    )
    parser.add_argument(
        "--frames-dir",
        "--rgb-images-dir",
        dest="frames_dir",
        default="train_frames",
        help=(
            "Frame root directory (absolute or relative to --road-dir). "
            "Expected layout: <frames-dir>/<video_name>/<frame>.jpg. "
            "Default: train_frames"
        ),
    )
    parser.add_argument(
        "--val-split",
        default=[3],
        help="List of split_ids to assign to val split. Can be a comma-separated string or a JSON list. Default: [3]",
    )
    parser.add_argument(
        "--train-split",
        default=[1, 2, 3],
        help="List of split_ids to assign to train split. Can be a comma-separated string or a JSON list. Default: [1,2,3]",
    )
    parser.add_argument(
        "--prepare-yolox-layout",
        action="store_true",
        help="Create train/ and val/ image folders for YOLOX under the output dir.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to populate YOLOX train/val image folders when preparing layout.",
    )
    parser.add_argument(
        "--match-emt",
        dest="match_emt",
        action="store_true",
        help="Map ROAD Waymo labels to EMT labels and ignore non-EMT traffic categories.",
    )
    parser.add_argument(
        "--no-match-emt",
        dest="match_emt",
        action="store_false",
        help="Keep raw ROAD labels and include all categories.",
    )

    parser.set_defaults(match_emt=True)
    return parser.parse_args()


def parse_split_ids(raw_split_ids, default_ids):
    if raw_split_ids is None:
        return set(default_ids)
    if isinstance(raw_split_ids, list):
        values = raw_split_ids
    else:
        raw = str(raw_split_ids).strip()
        if not raw:
            return set(default_ids)
        try:
            parsed = json.loads(raw)
            values = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            values = [part.strip() for part in raw.split(",") if part.strip()]

    parsed_ids = set()
    for value in values:
        try:
            parsed_ids.add(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid split id value: {value!r}. Use integers like 1,3 or [1,3]."
            ) from exc
    return parsed_ids


def split_from_entry(entry, train_split_ids, val_split_ids):
    split_ids = entry.get("split_ids", [])
    split_ids_lower = {str(item).lower() for item in split_ids}

    tagged_train_ids = set()
    tagged_val_ids = set()
    numeric_ids = set()

    for split_id in split_ids_lower:
        if split_id.startswith("train_") or split_id.startswith("val_"):
            prefix, suffix = split_id.split("_", 1)
            try:
                split_num = int(suffix)
            except ValueError:
                continue
            if prefix == "train":
                tagged_train_ids.add(split_num)
            else:
                tagged_val_ids.add(split_num)
            continue
        try:
            numeric_ids.add(int(split_id))
        except ValueError:
            continue

    if tagged_val_ids & val_split_ids:
        return "val"
    if tagged_train_ids & train_split_ids:
        return "train"
    if numeric_ids & val_split_ids:
        return "val"
    if numeric_ids & train_split_ids:
        return "train"
    if "val" in split_ids_lower:
        return "val"
    if "train" in split_ids_lower:
        return "train"
    return "train"


def resolve_image_path(frames_dir, frame_num):
    for width in (5, 6):
        candidate = frames_dir / f"{frame_num:0{width}d}.jpg"
        if candidate.exists():
            return candidate
    return None


def resolve_category_id(agent_labels, agent_value):
    if isinstance(agent_value, int):
        return agent_value
    return agent_labels.index(agent_value)


def convert_box(box, width, height):
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    else:
     print(f"--------------warning: box with large coordinates {box} for image size ({width}, {height}) - treating as absolute!!!!")
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def build_annotations(
    road_dir,
    annotations_path,
    frames_dir,
    match_emt=True,
    train_split_ids=None,
    val_split_ids=None,
):
    if train_split_ids is None:
        train_split_ids = {1, 2, 3}
    if val_split_ids is None:
        val_split_ids = {3}
    road = json.loads(annotations_path.read_text())
    agent_labels = road["all_agent_labels"]

    if match_emt:
        emt_names = [
            "Bus",
            "Car",
            "Cyclist",
            "Emergency_vehicle",
            "Large_vehicle",
            "Medium_vehicle",
            "Motorbike",
            "Pedestrian",
            "Small_motorised_vehicle",
        ]
        road_to_emt = {
            "Ped": "Pedestrian",
            "Car": "Car",
            "Cyc": "Cyclist",
            "Mobike": "Motorbike",
            "SmalVeh": "Small_motorised_vehicle",
            "MedVeh": "Medium_vehicle",
            "LarVeh": "Large_vehicle",
            "Bus": "Bus",
            "EmVeh": "Emergency_vehicle",
        }
        categories = [{"id": idx, "name": name} for idx, name in enumerate(emt_names)]
        category_name_to_id = {name: idx for idx, name in enumerate(emt_names)}
        print(
            "Matching EMT labels; ignoring TL/OthTL. "
            f"Output categories: {[c['name'] for c in categories]}"
        )
    else:
        categories = [{"id": idx, "name": name} for idx, name in enumerate(agent_labels)]
        category_name_to_id = {name: idx for idx, name in enumerate(agent_labels)}
        road_to_emt = None
        print(f"Found {len(categories)} categories: {[c['name'] for c in categories]}")

    split_payloads = {
        "train": {"images": [], "annotations": [], "categories": categories},
        "val": {"images": [], "annotations": [], "categories": categories},
    }
    image_ids = {"train": 0, "val": 0}
    ann_ids = {"train": 0, "val": 0}
    video_ids = {}
    track_ids = {}

    for video_idx, (video_name, entry) in enumerate(road["db"].items()):
        print(f"Processing video {video_idx}: {video_name}")
        split = split_from_entry(entry, train_split_ids, val_split_ids)
        print(f"  assigned to split: {split}")
        video_frames_dir = frames_dir / video_name
        if not video_frames_dir.exists():
            continue

        video_ids[video_name] = video_idx
        track_ids.setdefault(video_name, {})

        for frame_idx, frame_data in entry.get("frames", {}).items():
            frame_num = int(frame_idx)
            image_path = resolve_image_path(video_frames_dir, frame_num)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"warning: cannot read {image_path}")
                continue

            height, width = image.shape[:2]
            image_id = image_ids[split]
            split_payloads[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{video_name}/{image_path.name}",
                    "width": width,
                    "height": height,
                    "frame_id": frame_num,
                    "video_id": video_ids[video_name],
                }
            )

            for obj in frame_data.get("annos", {}).values():
                box = obj.get("box")
                if not box or len(box) != 4:
                    continue

                x1, y1, x2, y2 = convert_box(box, width, height)
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                agent_ids = obj.get("agent_ids") or []
                if not agent_ids:
                    continue
                raw_category_id = resolve_category_id(agent_labels, agent_ids[0])
                raw_category_name = agent_labels[raw_category_id]

                if match_emt:
                    mapped_name = road_to_emt.get(raw_category_name)
                    if mapped_name is None:
                        continue
                    category_id = category_name_to_id[mapped_name]
                else:
                    category_id = raw_category_id

                tube_uid = obj.get("tube_uid", f"{video_name}:{frame_num}:{ann_ids[split]}")
                video_tracks = track_ids[video_name]
                if tube_uid not in video_tracks:
                    video_tracks[tube_uid] = len(video_tracks) + 1

                split_payloads[split]["annotations"].append(
                    {
                        "id": ann_ids[split],
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0,
                        "track_id": video_tracks[tube_uid],
                    }
                )
                ann_ids[split] += 1

            image_ids[split] += 1

    return split_payloads


def write_annotations(output_dir, split_payloads):
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split, payload in split_payloads.items():
        (annotations_dir / f"{split}.json").write_text(json.dumps(payload))


def materialize_split_dir(destination_root, split, video_names, frames_dir, link_mode):
    split_dir = destination_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for video_name in video_names:
        src = frames_dir / video_name
        dst = split_dir / video_name
        if dst.exists() or dst.is_symlink():
            continue

        if link_mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            shutil.copytree(src, dst)


def prepare_yolox_layout(output_dir, frames_dir, split_payloads, link_mode):
    split_video_names = {
        split: sorted({image["file_name"].split("/", 1)[0] for image in payload["images"]})
        for split, payload in split_payloads.items()
    }
    for split, video_names in split_video_names.items():
        materialize_split_dir(output_dir, split, video_names, frames_dir, link_mode)


def main():
    args = parse_args()
    train_split_ids = parse_split_ids(args.train_split, default_ids=[1, 2, 3])
    val_split_ids = parse_split_ids(args.val_split, default_ids=[3])
    road_dir = Path(args.road_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (road_dir / "coco_annotations")
    annotations_path = Path(args.annotations_path)
    if not annotations_path.is_absolute():
        annotations_path = road_dir / annotations_path
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_absolute():
        frames_dir = road_dir / frames_dir
    if not annotations_path.exists():
        # Backward-compatible fallback for road_dir=datasets/road_uk.
        fallback_annotations = road_dir / "train" / args.annotations_path
        if fallback_annotations.exists():
            annotations_path = fallback_annotations
    if not frames_dir.exists():
        # Backward-compatible fallback for road_dir=datasets/road_uk.
        fallback_frames = road_dir / "train" / args.frames_dir
        if fallback_frames.exists():
            frames_dir = fallback_frames
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    print(f"Using train split ids: {sorted(train_split_ids)}")
    print(f"Using val split ids: {sorted(val_split_ids)}")
    print(f"Using annotations path: {annotations_path}")
    print(f"Using frames dir: {frames_dir}")
    split_payloads = build_annotations(
        road_dir,
        annotations_path=annotations_path,
        frames_dir=frames_dir,
        match_emt=args.match_emt,
        train_split_ids=train_split_ids,
        val_split_ids=val_split_ids,
    )
    write_annotations(output_dir, split_payloads)

    if args.prepare_yolox_layout:
        prepare_yolox_layout(output_dir, frames_dir, split_payloads, args.link_mode)

    for split in ("train", "val"):
        payload = split_payloads[split]
        print(
            f"{split}: {len(payload['images'])} images, "
            f"{len(payload['annotations'])} annotations"
        )
    print(f"annotations written to {output_dir / 'annotations'}")
    if args.prepare_yolox_layout:
        print(f"YOLOX image layout ready under {output_dir}")


if __name__ == "__main__":
    main()
