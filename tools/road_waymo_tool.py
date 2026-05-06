#!/usr/bin/env python3
"""
ROAD-Waymo annotation tool.

Supports:
  - Visualize annotated frames with bounding boxes
  - Print dataset statistics
  - Merge / remap classes and save a new annotation JSON

Built-in preset (--preset emt) matches the 3-class scheme used in yolox_emt.py:
  id=1  VulnerableRoadUser  ← Pedestrian, Cyclist
  id=2  Two-Wheeler         ← Motorbike, Small_motorised_vehicle
  id=3  Vehicle             ← Bus, Car, Large_vehicle, Medium_vehicle, Emergency_vehicle

Usage examples
--------------
  # Dataset statistics
  python tools/road_waymo_tool.py stats --split train

  # Visualize random frames (space/n = next, p = prev, q = quit)
  python tools/road_waymo_tool.py visualize --split train

  # Visualize one specific video sequence
  python tools/road_waymo_tool.py visualize --split train --video train_00444

  # Auto-play at ~5 fps
  python tools/road_waymo_tool.py visualize --split train --delay 200

  # Produce EMT 3-class JSON (preset)
  python tools/road_waymo_tool.py merge --split train --preset emt \\
      --out datasets/road_waymo/road_waymo_annotations/train_3class.json

  # Custom merge (drop everything not listed with --drop-unmapped)
  python tools/road_waymo_tool.py merge --split val \\
      --merge "Vehicle:Bus,Car,Large_vehicle,Medium_vehicle,Emergency_vehicle" \\
      --merge "VRU:Pedestrian,Cyclist" \\
      --drop-unmapped \\
      --out datasets/road_waymo/road_waymo_annotations/val_2class.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ─── paths ───────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "road_waymo"
ANN_DIR      = DATASET_ROOT / "road_waymo_annotations"
FRAMES_DIR   = DATASET_ROOT / "train_frames"

SPLIT_FILES = {
    "train": ANN_DIR / "train_class.json",
    "val":   ANN_DIR / "val_class.json",
    "train_3class": ANN_DIR / "train_3class.json",
    "val_3class": ANN_DIR / "val_3class.json",
}

# ─── built-in presets ────────────────────────────────────────────────────────

# Each preset: list of (new_name, supercategory, [old_class_names])
# New category IDs are assigned 1-based in list order (COCO convention).
PRESETS = {
    "emt": [
        ("VulnerableRoadUser", "person",  ["Pedestrian", "Cyclist"]),
        ("Two-Wheeler",        "vehicle", ["Motorbike", "Small_motorised_vehicle"]),
        ("Vehicle",            "vehicle", ["Bus", "Car", "Large_vehicle",
                                           "Medium_vehicle", "Emergency_vehicle"]),
    ],
}

# ─── colours (BGR) ───────────────────────────────────────────────────────────

PALETTE = [
    (0,   114, 189),
    (217, 83,  25),
    (237, 177, 32),
    (126, 47,  142),
    (119, 172, 48),
    (77,  190, 238),
    (162, 20,  47),
    (76,  76,  76),
    (153, 61,  113),
    (72,  176, 136),
]

def _color(idx: int):
    return PALETTE[idx % len(PALETTE)]

# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load_split(split: str):
    path = SPLIT_FILES.get(split)
    if path is None or not path.exists():
        sys.exit(f"[ERROR] Annotation file not found for split '{split}': {path}")
    print(f"Loading {path} …")
    with open(path) as f:
        data = json.load(f)
    return data


def build_index(data):
    id2img   = {img["id"]: img for img in data["images"]}
    img2anns = defaultdict(list)
    for ann in data["annotations"]:
        img2anns[ann["image_id"]].append(ann)
    id2cat   = {c["id"]: c["name"] for c in data["categories"]}
    return id2img, img2anns, id2cat

# ─── visualize ───────────────────────────────────────────────────────────────

def _draw_boxes(img, anns, id2cat):
    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cat_id = ann["category_id"]
        label  = id2cat.get(cat_id, str(cat_id))
        color  = _color(cat_id)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 2, y), color, -1)
        cv2.putText(img, label, (x + 1, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def cmd_visualize(args):
    data = load_split(args.split)
    id2img, img2anns, id2cat = build_index(data)

    images = list(data["images"])
    if args.video:
        images = [i for i in images if i["file_name"].startswith(args.video + "/")]
        if not images:
            sys.exit(f"[ERROR] No images found for video '{args.video}'")
        images.sort(key=lambda i: i["frame_id"])
        print(f"  Video '{args.video}': {len(images)} frames.")
    else:
        random.shuffle(images)
        print(f"  {len(images)} frames shuffled.")

    print("  Categories:", id2cat)
    print("  Keys: [space/n] next  [p] prev  [q] quit")

    idx = 0
    while 0 <= idx < len(images):
        img_info = images[idx]
        img_path = FRAMES_DIR / img_info["file_name"]
        if not img_path.exists():
            print(f"[WARN] Not found: {img_path}")
            idx += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read: {img_path}")
            idx += 1
            continue

        anns = img2anns[img_info["id"]]
        _draw_boxes(img, anns, id2cat)

        title = (f"[{idx+1}/{len(images)}]  "
                 f"{img_info['file_name']}  ({len(anns)} boxes)")
        cv2.namedWindow("road_waymo", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle("road_waymo", title)
        cv2.imshow("road_waymo", img)

        key = cv2.waitKey(args.delay if args.delay else 0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()

# ─── stats ────────────────────────────────────────────────────────────────────

def cmd_stats(args):
    data = load_split(args.split)
    id2cat = {c["id"]: c["name"] for c in data["categories"]}

    count_per_class = defaultdict(int)
    for ann in data["annotations"]:
        count_per_class[id2cat[ann["category_id"]]] += 1

    videos = {img["video_id"] for img in data["images"]}

    print(f"\n=== ROAD-Waymo [{args.split}] ===")
    print(f"  Videos      : {len(videos)}")
    print(f"  Images      : {len(data['images'])}")
    print(f"  Annotations : {len(data['annotations'])}")
    print(f"\n  Class distribution:")
    total = sum(count_per_class.values())
    for name, cnt in sorted(count_per_class.items(), key=lambda x: -x[1]):
        print(f"    {name:<28} {cnt:>9,}  ({100*cnt/total:.1f}%)")

# ─── merge ────────────────────────────────────────────────────────────────────

def _build_merge_plan_from_specs(specs):
    """Parse ['NewName:C1,C2', ...] → list of (new_name, supercategory, [old_names])."""
    plan = []
    for spec in specs:
        if ":" not in spec:
            sys.exit(f"[ERROR] Bad merge spec (expected 'NewName:C1,C2'): {spec}")
        new_name, members_str = spec.split(":", 1)
        members = [m.strip() for m in members_str.split(",") if m.strip()]
        plan.append((new_name.strip(), "object", members))
    return plan


def cmd_merge(args):
    if not args.preset and not args.merge:
        sys.exit("[ERROR] Provide --preset <name> or one or more --merge specs.")

    data    = load_split(args.split)
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    all_old = set(id2name.values())

    # build merge plan
    if args.preset:
        if args.preset not in PRESETS:
            sys.exit(f"[ERROR] Unknown preset '{args.preset}'. "
                     f"Available: {list(PRESETS.keys())}")
        plan = PRESETS[args.preset]
        if args.merge:
            print("[WARN] --preset and --merge both given; using preset only.")
    else:
        plan = _build_merge_plan_from_specs(args.merge)

    # validate old class names
    for new_name, supercat, old_names in plan:
        for old in old_names:
            if old not in all_old:
                sys.exit(f"[ERROR] Class '{old}' not found.\n"
                         f"  Available: {sorted(all_old)}")

    # old_name → (new_name, new_cat_id)  — 1-based COCO IDs
    old2new   = {}
    new_cats  = []
    for new_id_0based, (new_name, supercat, old_names) in enumerate(plan):
        new_id = new_id_0based + 1          # 1-based
        new_cats.append({"id": new_id, "name": new_name, "supercategory": supercat})
        for old in old_names:
            old2new[old] = (new_name, new_id)

    # handle unmapped classes
    unmapped = [n for n in all_old if n not in old2new]
    if unmapped and not args.drop_unmapped:
        next_id = len(new_cats) + 1
        for name in sorted(unmapped):
            new_cats.append({"id": next_id, "name": name, "supercategory": "object"})
            old2new[name] = (name, next_id)
            next_id += 1
    elif unmapped:
        print(f"[INFO] Dropping unmapped classes: {unmapped}")

    # summary
    print("\nClass mapping:")
    for old_id, old_name in sorted(id2name.items()):
        if old_name in old2new:
            new_name, new_id = old2new[old_name]
            print(f"  {old_name:<28} → [{new_id}] {new_name}")
        else:
            print(f"  {old_name:<28} → DROPPED")

    # remap annotations
    new_annotations = []
    dropped = 0
    for ann in data["annotations"]:
        old_name = id2name[ann["category_id"]]
        if old_name not in old2new:
            dropped += 1
            continue
        _, new_id = old2new[old_name]
        new_ann = dict(ann)
        new_ann["category_id"] = new_id
        new_annotations.append(new_ann)

    out_data = {
        "images":      data["images"],
        "annotations": new_annotations,
        "categories":  new_cats,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f)

    print(f"\nSaved → {out_path}")
    print(f"  New categories  : {[(c['id'], c['name']) for c in new_cats]}")
    print(f"  Annotations kept: {len(new_annotations):,}  (dropped {dropped:,})")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="ROAD-Waymo annotation reader / visualizer / class merger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # visualize
    vis = sub.add_parser("visualize", aliases=["vis", "v"],
                         help="Show annotated frames")
    vis.add_argument("--split",  default="train", choices=["train", "val"])
    vis.add_argument("--video",  default=None,
                     help="Limit to one video sequence, e.g. train_00444")
    vis.add_argument("--delay",  type=int, default=0,
                     help="Auto-advance delay ms (0 = wait for keypress)")
    vis.add_argument("--seed",   type=int, default=42)

    # stats
    st = sub.add_parser("stats", aliases=["s"],
                        help="Print dataset statistics")
    st.add_argument("--split", default="train", choices=["train", "val","train_3class", "val_3class"])

    # merge
    mg = sub.add_parser("merge", aliases=["m"],
                        help="Remap / merge classes and write new annotation JSON")
    mg.add_argument("--split",   default="train", choices=["train", "val"])
    mg.add_argument("--preset",  default=None, choices=list(PRESETS.keys()),
                     help="Use a built-in class mapping (e.g. 'emt' for 3-class)")
    mg.add_argument("--merge",   action="append", metavar="NewName:C1,C2,...",
                    help="Custom merge spec (repeatable); ignored when --preset is set")
    mg.add_argument("--drop-unmapped", action="store_true",
                    help="Drop annotations whose class is not covered by any merge spec")
    mg.add_argument("--out",     required=True, metavar="PATH",
                    help="Output JSON path")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.cmd in ("visualize", "vis", "v"):
        random.seed(args.seed)
        cmd_visualize(args)
    elif args.cmd in ("stats", "s"):
        cmd_stats(args)
    elif args.cmd in ("merge", "m"):
        cmd_merge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
