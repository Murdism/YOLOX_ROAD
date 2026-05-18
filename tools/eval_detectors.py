#!/usr/bin/env python3
"""
Evaluate one or more detectors against a COCO-format ground-truth file.

Metrics reported per class and overall (micro-averaged):
  Precision ↑   Recall ↑   F1 ↑   FP ↓   FN ↓

Usage — named detectors (recommended)
--------------------------------------
python tools/eval_detectors.py --split test --detectors emt waymo emt_waymo

Usage — raw files (custom / ad-hoc)
-------------------------------------
python tools/eval_detectors.py \
    --gt   datasets/EMT/annotations/detections_new/test_3class.json \
    --pred results/my_det_test_preds.json \
    --name "My Detector" \
    --iou  0.5 --conf 0.3

Generate prediction JSON files with:
  python tools/dump_predictions.py --detector emt   --split test
  python tools/dump_predictions.py --detector waymo --split test
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Registry — edit here to add / rename detectors
# ---------------------------------------------------------------------------

# Prediction JSON path template.  {split} is replaced with "train" or "test".
# Generate these files with tools/dump_predictions.py
DETECTOR_REGISTRY = {
    "emt":       "results/emt_{split}_preds.json",
    "waymo":     "results/waymo_{split}_preds.json",
    "emt_waymo": "results/emt_waymo_{split}_preds.json",
}

DETECTOR_DISPLAY = {
    "emt":       "COCO → EMT",
    "waymo":     "COCO → Waymo",
    "emt_waymo": "Waymo → EMT",
}

# Ground-truth annotation files per split
GT_PATHS = {
    "test":  "datasets/EMT/annotations/detections_new/test_3class.json",
    "train": "datasets/EMT/annotations/detections_new/train_3class.json",
}

ALL_SPLITS = ("train", "test")  # evaluated and summed when --split all


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """(G,4) × (P,4) xyxy → (G,P) IoU"""
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]))

    ix1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    iy1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    ix2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    iy2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    gt_area   = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gt(gt_path: str):
    """Returns (gt_index, categories).

    gt_index   : {image_id: {cat_id: ndarray (N,4) xyxy}}
    categories : {cat_id: name}
    """
    with open(gt_path) as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    raw: dict = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        raw[ann["image_id"]][ann["category_id"]].append(ann["bbox"])

    gt_index = {
        img_id: {
            cat_id: xywh_to_xyxy(np.array(boxes, dtype=np.float32))
            for cat_id, boxes in cats.items()
        }
        for img_id, cats in raw.items()
    }
    return gt_index, categories


def load_preds(pred_path: str, conf_thr: float):
    """Returns {image_id: {cat_id: [(score, box_xyxy), ...]}} sorted desc by score."""
    with open(pred_path) as f:
        preds = json.load(f)

    index: dict = defaultdict(lambda: defaultdict(list))
    for det in preds:
        if det["score"] < conf_thr:
            continue
        box = xywh_to_xyxy(np.array([det["bbox"]], dtype=np.float32))[0]
        index[det["image_id"]][det["category_id"]].append((det["score"], box))

    for img_id in index:
        for cat_id in index[img_id]:
            index[img_id][cat_id].sort(key=lambda x: -x[0])

    return index


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_image_category(gt_boxes, pred_entries, iou_thr):
    """Greedy highest-confidence-first matching → (tp, fp, fn)."""
    n_gt, n_pred = len(gt_boxes), len(pred_entries)
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    pred_boxes   = np.stack([e[1] for e in pred_entries])
    iou          = iou_matrix(gt_boxes, pred_boxes)
    gt_matched   = np.zeros(n_gt,   dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)

    for p in range(n_pred):
        g = int(np.argmax(iou[:, p]))
        if iou[g, p] >= iou_thr and not gt_matched[g]:
            gt_matched[g] = pred_matched[p] = True

    tp = int(pred_matched.sum())
    return tp, n_pred - tp, int((~gt_matched).sum())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(gt_index, preds_index, categories, iou_thr):
    """Returns {cat_id: {"tp": int, "fp": int, "fn": int}} for one split."""
    per_class = {cat_id: {"tp": 0, "fp": 0, "fn": 0} for cat_id in categories}

    for img_id in set(gt_index) | set(preds_index):
        gt_cats   = gt_index.get(img_id, {})
        pred_cats = preds_index.get(img_id, {})

        for cat_id in set(gt_cats) | set(pred_cats):
            if cat_id not in per_class:
                per_class[cat_id] = {"tp": 0, "fp": 0, "fn": 0}
            tp, fp, fn = match_image_category(
                gt_cats.get(cat_id, np.empty((0, 4), dtype=np.float32)),
                pred_cats.get(cat_id, []),
                iou_thr,
            )
            per_class[cat_id]["tp"] += tp
            per_class[cat_id]["fp"] += fp
            per_class[cat_id]["fn"] += fn

    return per_class


def add_per_class(a, b):
    """Sum TP/FP/FN across two per_class dicts (used when combining splits)."""
    return {
        k: {
            "tp": a.get(k, {}).get("tp", 0) + b.get(k, {}).get("tp", 0),
            "fp": a.get(k, {}).get("fp", 0) + b.get(k, {}).get("fp", 0),
            "fn": a.get(k, {}).get("fn", 0) + b.get(k, {}).get("fn", 0),
        }
        for k in set(a) | set(b)
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_per_detector_table(per_class, categories, name):
    rows = []
    tot_tp = tot_fp = tot_fn = 0

    for cat_id in sorted(per_class):
        d = per_class[cat_id]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p, r, f1   = _prf(tp, fp, fn)
        rows.append([categories.get(cat_id, f"cls_{cat_id}"),
                     f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", fp, fn, tp])
        tot_tp += tp; tot_fp += fp; tot_fn += fn

    p, r, f1 = _prf(tot_tp, tot_fp, tot_fn)
    rows.append(["OVERALL (micro)",
                 f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", tot_fp, tot_fn, tot_tp])

    sep = "─" * 68
    return (
        f"\n{sep}\n Detector : {name}\n{sep}\n"
        + tabulate(rows,
                   headers=["Class", "Precision↑", "Recall↑", "F1↑", "FP↓", "FN↓", "TP"],
                   tablefmt="pipe", numalign="right")
    )


def build_comparison_table(all_results, categories):
    det_names = [r["name"] for r in all_results]
    cat_ids   = sorted(categories)
    header    = ["Class"] + det_names

    def _rows(pick):
        rows = []
        for cat_id in cat_ids:
            cname = categories.get(cat_id, f"cls_{cat_id}")
            vals  = []
            for res in all_results:
                d = res["per_class"].get(cat_id, {"tp": 0, "fp": 0, "fn": 0})
                vals.append(f"{pick(*_prf(d['tp'], d['fp'], d['fn'])):.4f}")
            rows.append([cname] + vals)

        ovals = []
        for res in all_results:
            tp = sum(res["per_class"].get(c, {}).get("tp", 0) for c in cat_ids)
            fp = sum(res["per_class"].get(c, {}).get("fp", 0) for c in cat_ids)
            fn = sum(res["per_class"].get(c, {}).get("fn", 0) for c in cat_ids)
            ovals.append(f"{pick(*_prf(tp, fp, fn)):.4f}")
        rows.append(["OVERALL (micro)"] + ovals)
        return rows

    sep  = "─" * 68
    esep = "═" * 68
    lines = [f"\n{esep}\n COMPARISON ACROSS DETECTORS\n{esep}"]
    for label, pick in [
        ("F1 Score ↑",  lambda p, r, f: f),
        ("Precision ↑", lambda p, r, f: p),
        ("Recall ↑",    lambda p, r, f: r),
    ]:
        lines.append(f"\n{sep}\n {label}\n{sep}")
        lines.append(tabulate(_rows(pick), headers=header, tablefmt="pipe", numalign="right"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_gt_paths(split):
    if split == "all":
        return [GT_PATHS[s] for s in ALL_SPLITS]
    if split not in GT_PATHS:
        raise ValueError(f"Unknown split '{split}'. Choose: {list(GT_PATHS)} or 'all'.")
    return [GT_PATHS[split]]


def resolve_pred_paths(detector_key, split):
    """Returns list of (gt_path, pred_path) pairs for the given split."""
    template = DETECTOR_REGISTRY[detector_key]
    if split == "all":
        return [(GT_PATHS[s], template.format(split=s)) for s in ALL_SPLITS]
    return [(GT_PATHS[split], template.format(split=split))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(
        description="Detector evaluation: Precision / Recall / F1 / FP / FN",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    named = parser.add_argument_group("Named-detector mode")
    named.add_argument(
        "--split", default="test",
        choices=list(GT_PATHS) + ["all"],
        help="Split to evaluate: train | test | all  (default: test).",
    )
    named.add_argument(
        "--detectors", nargs="+", metavar="NAME",
        choices=list(DETECTOR_REGISTRY),
        help=f"Named detector(s): {list(DETECTOR_REGISTRY)}",
    )

    raw = parser.add_argument_group("Raw-file mode (custom paths)")
    raw.add_argument("--gt",   help="COCO ground-truth JSON (overrides --split).")
    raw.add_argument("--pred", nargs="+", help="COCO detection-result JSON file(s).")
    raw.add_argument("--name", nargs="*", help="Label per --pred file.")

    parser.add_argument("--iou",  type=float, default=0.5,
                        help="IoU threshold for TP/FP matching (default: 0.5).")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Min confidence to keep a prediction (default: 0.3).")
    parser.add_argument("--save", default=None,
                        help="Optional path to write results as plain text.")
    return parser


def main():
    args = make_parser().parse_args()

    # ── build job list: [(gt_pred_pairs, display_name), ...] ───────────────
    if args.detectors:
        jobs = []
        for key in args.detectors:
            pairs   = resolve_pred_paths(key, args.split)
            display = DETECTOR_DISPLAY.get(key, key)
            jobs.append((pairs, display))
        split_label = args.split

    elif args.gt and args.pred:
        names = args.name if args.name else [Path(p).stem for p in args.pred]
        if len(names) != len(args.pred):
            raise ValueError("--name must have the same number of entries as --pred")
        jobs = [([(args.gt, p)], n) for p, n in zip(args.pred, names)]
        split_label = "custom"

    else:
        make_parser().error(
            "Provide --detectors (named mode) or both --gt and --pred (raw mode)."
        )

    print(f"\nSplit        : {split_label}")
    print(f"IoU threshold: {args.iou}   Conf threshold: {args.conf}\n")

    # ── evaluate ────────────────────────────────────────────────────────────
    all_results  = []
    output_parts = []
    categories   = None

    for gt_pred_pairs, det_name in jobs:
        combined = {}
        ok = False

        for gt_path, pred_path in gt_pred_pairs:
            if not Path(pred_path).exists():
                det_key = next(
                    (k for k, v in DETECTOR_REGISTRY.items()
                     if pred_path in (v.format(split=s) for s in ALL_SPLITS + ("test", "train"))),
                    "?"
                )
                sp = next(
                    (s for s in ALL_SPLITS + ("test", "train")
                     if DETECTOR_REGISTRY.get(det_key, "").format(split=s) == pred_path),
                    "?"
                )
                print(f"  [skip] {pred_path} not found.\n"
                      f"         Generate with: python tools/dump_predictions.py "
                      f"--detector {det_key} --split {sp}\n")
                continue

            gt_index, cats = load_gt(gt_path)
            if categories is None:
                categories = cats
            preds_index  = load_preds(pred_path, args.conf)
            split_result = evaluate_split(gt_index, preds_index, cats, args.iou)
            combined     = add_per_class(combined, split_result)
            ok = True
            n_gt_imgs = len(gt_index)
            print(f"  [{Path(pred_path).name}]  GT images: {n_gt_imgs}  "
                  f"categories: {list(cats.values())}")

        if not ok or not combined:
            continue

        all_results.append({"name": det_name, "per_class": combined})
        block = build_per_detector_table(combined, categories, det_name)
        print(block)
        output_parts.append(block)

    if not all_results:
        print("\nNo results — check that prediction files exist.")
        return

    if len(all_results) > 1:
        comp = build_comparison_table(all_results, categories)
        print(comp)
        output_parts.append(comp)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text("\n".join(output_parts))
        print(f"\nResults saved → {args.save}")


if __name__ == "__main__":
    main()
