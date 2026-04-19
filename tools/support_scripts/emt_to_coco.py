#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2

SUPERCLASS_NAME_MAP = {
    "Motorbike": "Motorbike",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Car": "Vehicle",
    "Bus": "Vehicle",
    "Medium_vehicle": "Vehicle",
    "Large_vehicle": "Vehicle",
    "Emergency_vehicle": "Vehicle",
    "Small_motorised_vehicle": "Vehicle",
    "Small_motorized_vehicle": "Vehicle",
}
SUPERCLASS_ORDER = ("Motorbike", "Pedestrian", "Vehicle", "Cyclist")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EMT KITTI-style tracking labels into COCO JSON for YOLOX."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/EMT",
        help="EMT dataset root containing frames/ and annotations/.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help=(
            "Optional label directory (absolute or relative to dataset root). "
            "If omitted, the script auto-detects common EMT KITTI label paths."
        ),
    )
    parser.add_argument(
        "--labels-subdir",
        default="labels_full",
        help=(
            "Legacy subdirectory under --annotation-dir to read labels from "
            "when --labels-dir is not provided."
        ),
    )
    parser.add_argument(
        "--annotation-dir",
        default="annotations/detections_new",
        help="Directory under dataset root where output JSON files are written.",
    )
    parser.add_argument(
        "--train-images-dir",
        default="frames",
        help="Directory under dataset root containing training frame folders.",
    )
    parser.add_argument(
        "--val-images-dir",
        default="frames",
        help="Directory under dataset root containing validation frame folders.",
    )
    parser.add_argument(
        "--train-output",
        default="train_superclass.json",
        help="COCO JSON filename for the training split.",
    )
    parser.add_argument(
        "--val-output",
        default="test_superclass.json",
        help="COCO JSON filename for the validation split.",
    )
    parser.add_argument(
        "--label-level",
        choices=("class", "superclass"),
        default="superclass",
        help=(
            "Use original EMT class labels or map to 4 superclasses: "
            "Motorbike, pedestrian, vehicle, cyclist."
        ),
    )
    return parser.parse_args()


def normalize_class_name(name):
    return name.strip().replace("-", "_").replace(" ", "_").lower()


def map_class_name(class_name, label_level):
    if label_level == "class":
        return class_name

    mapped = SUPERCLASS_NAME_MAP.get(class_name)
    if mapped is not None:
        return mapped

    normalized = normalize_class_name(class_name)
    for source, target in SUPERCLASS_NAME_MAP.items():
        if normalize_class_name(source) == normalized:
            return target
    return None


def build_categories(label_dir, split_videos, label_level):
    class_names = set()
    for video_name in split_videos:
        label_path = label_dir / f"{video_name}.txt"
        if not label_path.exists():
            continue
        with label_path.open() as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 3:
                    mapped_name = map_class_name(parts[2], label_level)
                    if mapped_name is not None:
                        class_names.add(mapped_name)

    categories = []
    class_to_id = {}
    if label_level == "superclass":
        ordered_names = [name for name in SUPERCLASS_ORDER if name in class_names]
        ordered_names.extend(sorted(class_names - set(ordered_names)))
    else:
        ordered_names = sorted(class_names)

    for category_id, class_name in enumerate(ordered_names, start=1):
        class_to_id[class_name] = category_id
        categories.append({"id": category_id, "name": class_name})
    return categories, class_to_id


def build_frame_index(video_dir):
    frame_index = {}
    for image_path in sorted(video_dir.glob("*.jpg")):
        frame_index[int(image_path.stem)] = image_path
    return frame_index


def read_video_image_size(frame_index, video_dir):
    if not frame_index:
        raise ValueError(f"no frames found in {video_dir}")

    sample_path = next(iter(frame_index.values()))
    image = cv2.imread(str(sample_path))
    if image is None:
        raise ValueError(f"cannot read image {sample_path}")
    return image.shape[:2]


def parse_label_file(label_path):
    frame_to_objects = {}
    if not label_path.exists():
        return frame_to_objects

    with label_path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                class_name = parts[2]
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
            except ValueError as exc:
                raise ValueError(f"invalid line {line_num} in {label_path}") from exc

            frame_to_objects.setdefault(frame_id, []).append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    return frame_to_objects


def load_split_videos_from_coco(json_path):
    if not json_path.exists():
        return set()

    payload = json.loads(json_path.read_text())
    video_names = set()
    for image in payload.get("images", []):
        file_name = image.get("file_name", "")
        if "/" in file_name:
            video_names.add(file_name.split("/", 1)[0])
    return video_names


def list_video_dirs(image_root):
    if not image_root.exists():
        return set()
    return {path.name for path in image_root.iterdir() if path.is_dir()}


def resolve_split_video_dirs(train_root, val_root, annotation_root, train_output, val_output):
    train_videos = list_video_dirs(train_root)
    val_videos = list_video_dirs(val_root)

    if train_videos and val_videos and train_root.resolve() != val_root.resolve():
        train_videos -= val_videos
        return train_videos, val_videos, train_root, val_root

    shared_root = train_root if train_videos else val_root
    if not shared_root.exists():
        raise ValueError(
            f"no image folders found: train={train_root} val={val_root}"
        )

    # Prefer canonical split hints when available.
    mapped_train_videos = load_split_videos_from_coco(annotation_root / "train.json")
    mapped_val_videos = load_split_videos_from_coco(annotation_root / "test.json")
    if not (mapped_train_videos or mapped_val_videos):
        mapped_train_videos = load_split_videos_from_coco(annotation_root / train_output)
        mapped_val_videos = load_split_videos_from_coco(annotation_root / val_output)
    if mapped_train_videos or mapped_val_videos:
        return mapped_train_videos, mapped_val_videos, shared_root, shared_root

    return list_video_dirs(shared_root), set(), shared_root, shared_root


def resolve_label_dir(args, dataset_root):
    if args.labels_dir is not None:
        labels_dir = Path(args.labels_dir)
        if not labels_dir.is_absolute():
            labels_dir = dataset_root / labels_dir
        return labels_dir

    candidates = [
        dataset_root / args.annotation_dir / args.labels_subdir,
        dataset_root / "annotations" / "tracking" / "kitti",
        dataset_root / "annotations" / "tracking" / "gmot",
        dataset_root / "emt_annotations" / "labels_full",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def build_split_payload(
    split_name, video_names, image_root, label_dir, class_to_id, label_level
):
    payload = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    ann_id = 1
    track_id_map = {}

    for video_id, video_name in enumerate(sorted(video_names), start=1):
        video_dir = image_root / video_name
        if not video_dir.exists():
            continue

        frame_index = build_frame_index(video_dir)
        height, width = read_video_image_size(frame_index, video_dir)
        frame_to_objects = parse_label_file(label_dir / f"{video_name}.txt")
        track_id_map[video_name] = {}

        for frame_num, image_path in frame_index.items():
            payload["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{video_name}/{image_path.name}",
                    "width": width,
                    "height": height,
                    "frame_id": frame_num,
                    "video_id": video_id,
                }
            )

            for obj in frame_to_objects.get(frame_num, []):
                mapped_class_name = map_class_name(obj["class_name"], label_level)
                if mapped_class_name is None:
                    continue
                if mapped_class_name not in class_to_id:
                    continue

                x1, y1, x2, y2 = clip_bbox(obj["bbox"], width, height)
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                video_tracks = track_id_map[video_name]
                original_track_id = obj["track_id"]
                if original_track_id not in video_tracks:
                    video_tracks[original_track_id] = len(video_tracks) + 1

                payload["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_to_id[mapped_class_name],
                        "bbox": [x1, y1, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0,
                        "track_id": video_tracks[original_track_id],
                    }
                )
                ann_id += 1

            image_id += 1

    return payload


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    annotation_root = dataset_root / args.annotation_dir
    label_dir = resolve_label_dir(args, dataset_root)
    if not label_dir.exists():
        raise ValueError(f"label directory does not exist: {label_dir}")

    train_root = dataset_root / args.train_images_dir
    val_root = dataset_root / args.val_images_dir

    train_videos, val_videos, train_root, val_root = resolve_split_video_dirs(
        train_root, val_root, annotation_root, args.train_output, args.val_output
    )

    categories, class_to_id = build_categories(
        label_dir, train_videos | val_videos, args.label_level
    )

    train_payload = build_split_payload(
        "train", train_videos, train_root, label_dir, class_to_id, args.label_level
    )
    val_payload = build_split_payload(
        "val", val_videos, val_root, label_dir, class_to_id, args.label_level
    )
    train_payload["categories"] = categories
    val_payload["categories"] = categories

    annotation_root.mkdir(parents=True, exist_ok=True)
    train_path = annotation_root / args.train_output
    val_path = annotation_root / args.val_output
    train_path.write_text(json.dumps(train_payload))
    val_path.write_text(json.dumps(val_payload))

    print(
        f"train: {len(train_payload['images'])} images, "
        f"{len(train_payload['annotations'])} annotations"
    )
    print(
        f"val: {len(val_payload['images'])} images, "
        f"{len(val_payload['annotations'])} annotations"
    )
    print(f"label level: {args.label_level}")
    print(f"labels read from: {label_dir}")
    print(f"categories: {len(categories)} -> {[cat['name'] for cat in categories]}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
