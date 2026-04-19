#!/usr/bin/env python3
"""Print per-video bbox counts for each class from EMT COCO JSON files."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Show number of bboxes for each class in each video."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=repo_root / "datasets/EMT/annotations/detections_new",
        help="Directory that contains COCO annotation JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Split names to read. Each split maps to <split>.json.",
    )
    return parser.parse_args()


def get_video_name(image: dict) -> str:
    file_name = image.get("file_name", "")
    if isinstance(file_name, str) and "/" in file_name:
        return file_name.split("/", 1)[0]

    video_id = image.get("video_id")
    if video_id is not None:
        return f"video_{video_id}"
    return "video_unknown"


def video_sort_key(video_name: str) -> Tuple[int, object]:
    tail = video_name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return (0, int(tail))
    return (1, video_name)


def load_split(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a top-level JSON object.")
    return data


def count_video_class_boxes(data: dict) -> Tuple[Dict[int, str], Dict[str, Counter]]:
    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    category_id_to_name: Dict[int, str] = {}
    for c in categories:
        if isinstance(c, dict) and "id" in c:
            category_id_to_name[c["id"]] = c.get("name", f"class_{c['id']}")

    image_by_id = {}
    for image in images:
        if isinstance(image, dict) and image.get("id") is not None:
            image_by_id[image["id"]] = image

    video_class_counts: Dict[str, Counter] = defaultdict(Counter)
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        # print(f"Processing annotation: {ann}")
        # print(ann.get("track_id_str")[0])

        
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id is None or category_id is None:
            continue

        image = image_by_id.get(image_id)
        if image is None:
            continue

        video_name = get_video_name(image)
        video_class_counts[video_name][category_id] += 1

    return category_id_to_name, video_class_counts


def print_report(split: str, category_id_to_name: Dict[int, str], video_class_counts: Dict[str, Counter]) -> None:
    print(f"\n[{split}] videos: {len(video_class_counts)}")
    if not video_class_counts:
        return

    ordered_known_ids = sorted(category_id_to_name.keys())
    for video_name in sorted(video_class_counts.keys(), key=video_sort_key):
        counts = video_class_counts[video_name]
        total_boxes = sum(counts.values())

        parts: List[str] = [f"total={total_boxes}"]
        for cid in ordered_known_ids:
            cname = category_id_to_name[cid]
            parts.append(f"{cname}={counts.get(cid, 0)}")

        unknown_ids = sorted(cid for cid in counts if cid not in category_id_to_name)
        for cid in unknown_ids:
            parts.append(f"class_{cid}={counts[cid]}")

        print(f"{video_name}\t" + "\t".join(parts))


def iter_split_paths(annotations_dir: Path, splits: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        yield split, annotations_dir / f"{split}.json"


def main() -> int:
    args = parse_args()
    annotations_dir: Path = args.annotations_dir
    splits: List[str] = args.splits

    status = 0
    for split, split_path in iter_split_paths(annotations_dir, splits):
        if not split_path.exists():
            print(f"\n[{split}] missing file: {split_path}")
            status = 1
            continue

        try:
            data = load_split(split_path)
            category_id_to_name, video_class_counts = count_video_class_boxes(data)
            print_report(split, category_id_to_name, video_class_counts)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"\n[{split}] error: {exc}")
            status = 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
