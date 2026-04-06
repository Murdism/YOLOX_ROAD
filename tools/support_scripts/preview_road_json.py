#!/usr/bin/env python3

import argparse
import json
from itertools import islice
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the first N entries from ROAD Waymo JSON."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("datasets/road_uk/train/annotations/road_trainval_v1.0.json"), #road_waymo_trainval_v1.0.json"),
        help="Path to road_waymo_trainval_v1.0.json",
    )
    parser.add_argument(
        "-x",
        "--count",
        type=int,
        default=500,
        help="Number of entries to print.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full JSON content for each selected entry.",
    )
    return parser.parse_args()


def summarize_video_entry(video_name, entry, idx):
    frames = entry.get("frames", {})
    frame_keys = sorted(int(k) for k in frames.keys()) if frames else []
    split_ids = entry.get("split_ids", [])
    numf = entry.get("numf")
    first_frame = frame_keys[0] if frame_keys else None
    last_frame = frame_keys[-1] if frame_keys else None
    sample_keys = frame_keys[:5]

    print(f"[{idx}] {video_name}")
    print(f"  split_ids: {split_ids}")
    print(f"  numf: {numf}")
    print(f"  annotated_frame_entries: {len(frame_keys)}")
    print(f"  frame_range: {first_frame} -> {last_frame}")
    print(f"  first_frame_keys: {sample_keys}")

    if frame_keys:
        first_frame_data = frames[str(first_frame)]
        annos = first_frame_data.get("annos", {})
        print(f"  first_frame_anno_count: {len(annos)}")


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if not args.json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")

    data = json.loads(args.json_path.read_text())
    print(f"Loaded: {args.json_path}")
    print(f"Top-level keys: {list(data.keys())}")

    db = data.get("db")
    if not isinstance(db, dict):
        raise ValueError("Expected key 'db' to be a dictionary in ROAD JSON.")

    print(f"Total db entries: {len(db)}")
    print(f"Printing first {args.count} entries...\n")

    for idx, (video_name, entry) in enumerate(islice(db.items(), args.count), start=1):
        if args.full:
            print(f"[{idx}] {video_name}")
            print(json.dumps(entry, indent=2))
        else:
            summarize_video_entry(video_name, entry, idx)
        print()


if __name__ == "__main__":
    main()
