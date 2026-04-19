#!/usr/bin/env python3
"""Read EMT train/test COCO annotation JSON files and print quick stats."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read EMT train/test JSON annotations and print summary stats."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("datasets/EMT/annotations/detections_new"),
        help="Directory that contains train.json and test.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_superclass", "test_superclass"],
        help="Split names to read (each split maps to <split>.json).",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print one sample image and one sample annotation per split.",
    )
    return parser.parse_args()


def _safe_len(obj: object) -> int:
    return len(obj) if hasattr(obj, "__len__") else 0


def _build_category_maps(categories: List[dict]) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        cat_id = category.get("id")
        cat_name = category.get("name")
        if isinstance(cat_id, int):
            id_to_name[cat_id] = str(cat_name) if cat_name is not None else "<unnamed>"
    return id_to_name


def summarize_annotations(data: dict) -> Dict[str, object]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_ids = {
        image.get("id")
        for image in images
        if isinstance(image, dict) and image.get("id") is not None
    }

    per_category = Counter()
    missing_image_refs = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        category_id = annotation.get("category_id")
        if category_id is not None:
            per_category[category_id] += 1

        image_id = annotation.get("image_id")
        if image_id is not None and image_id not in image_ids:
            missing_image_refs += 1

    category_id_to_name = _build_category_maps(categories)
    unknown_category_ids = sorted(
        cid for cid in per_category.keys() if cid not in category_id_to_name
    )

    return {
        "num_images": _safe_len(images),
        "num_annotations": _safe_len(annotations),
        "num_categories": _safe_len(categories),
        "category_id_to_name": category_id_to_name,
        "per_category": per_category,
        "unknown_category_ids": unknown_category_ids,
        "missing_image_refs": missing_image_refs,
        "sample_image": images[0] if images else None,
        "sample_annotation": annotations[0] if annotations else None,
    }


def print_split_summary(split: str, summary: Dict[str, object], show_sample: bool) -> None:
    print(f"\n[{split}]")
    print(f"images:       {summary['num_images']}")
    print(f"annotations:  {summary['num_annotations']}")
    print(f"categories:   {summary['num_categories']}")
    print(f"bad refs:     {summary['missing_image_refs']} (annotations with missing image_id)")

    category_id_to_name: Dict[int, str] = summary["category_id_to_name"]  # type: ignore[assignment]
    per_category: Counter = summary["per_category"]  # type: ignore[assignment]
    unknown_category_ids: List[int] = summary["unknown_category_ids"]  # type: ignore[assignment]

    if category_id_to_name:
        print("per-category annotations:")
        for category_id in sorted(category_id_to_name.keys()):
            name = category_id_to_name[category_id]
            count = per_category.get(category_id, 0)
            print(f"  - {category_id:>3} {name:<20} {count}")
    else:
        print("per-category annotations: <no categories>")

    if unknown_category_ids:
        print(f"unknown category ids in annotations: {unknown_category_ids}")

    if show_sample:
        print("\nsample image:")
        print(json.dumps(summary["sample_image"], indent=2, ensure_ascii=True))
        print("sample annotation:")
        print(json.dumps(summary["sample_annotation"], indent=2, ensure_ascii=True))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a top-level JSON object.")
    return data


def iter_split_paths(annotations_dir: Path, splits: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        yield split, annotations_dir / f"{split}.json"


def main() -> int:
    args = parse_args()
    annotations_dir: Path = args.annotations_dir
    splits: List[str] = args.splits

    if not annotations_dir.exists():
        print(f"error: directory not found: {annotations_dir}")
        return 1

    status = 0
    for split, json_path in iter_split_paths(annotations_dir, splits):
        if not json_path.exists():
            print(f"\n[{split}]")
            print(f"error: file not found: {json_path}")
            status = 1
            continue

        try:
            data = load_json(json_path)
            summary = summarize_annotations(data)
            print_split_summary(split, summary, args.show_sample)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"\n[{split}]")
            print(f"error reading {json_path}: {exc}")
            status = 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
