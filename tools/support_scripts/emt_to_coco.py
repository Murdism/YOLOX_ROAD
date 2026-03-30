#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert EMT KITTI-style tracking labels into COCO JSON for YOLOX."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/emt",
        help="EMT dataset root containing frames/, test_frames/, and emt_annotations/.",
    )
    parser.add_argument(
        "--labels-subdir",
        default="labels_full",
        help="Annotation subdirectory inside emt_annotations to read from.",
    )
    parser.add_argument(
        "--annotation-dir",
        default="emt_annotations",
        help="Directory under dataset root where output JSON files are written.",
    )
    parser.add_argument(
        "--train-images-dir",
        default="frames",
        help="Directory under dataset root containing training frame folders.",
    )
    parser.add_argument(
        "--val-images-dir",
        default="test_frames",
        help="Directory under dataset root containing validation frame folders.",
    )
    parser.add_argument(
        "--train-output",
        default="train.json",
        help="COCO JSON filename for the training split.",
    )
    parser.add_argument(
        "--val-output",
        default="test.json",
        help="COCO JSON filename for the validation split.",
    )
    return parser.parse_args()


def build_categories(label_dir, split_videos):
    class_names = set()
    for video_name in split_videos:
        label_path = label_dir / f"{video_name}.txt"
        if not label_path.exists():
            continue
        with label_path.open() as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 3:
                    class_names.add(parts[2])

    categories = []
    class_to_id = {}
    for category_id, class_name in enumerate(sorted(class_names), start=1):
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


def build_split_payload(split_name, video_names, image_root, label_dir, class_to_id):
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
                if obj["class_name"] not in class_to_id:
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
                        "category_id": class_to_id[obj["class_name"]],
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
    label_dir = annotation_root / args.labels_subdir
    train_root = dataset_root / args.train_images_dir
    val_root = dataset_root / args.val_images_dir

    train_videos = {path.name for path in train_root.iterdir() if path.is_dir()}
    val_videos = {path.name for path in val_root.iterdir() if path.is_dir()}
    train_videos -= val_videos

    categories, class_to_id = build_categories(label_dir, train_videos | val_videos)

    train_payload = build_split_payload(
        "train", train_videos, train_root, label_dir, class_to_id
    )
    val_payload = build_split_payload(
        "val", val_videos, val_root, label_dir, class_to_id
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
    print(f"categories: {len(categories)} -> {[cat['name'] for cat in categories]}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
