#!/usr/bin/env python3
"""
Compute PR curves, AP@IoU, and COCO mAP@0.5:0.95 for one or more detectors.
Produces a combined figure (per-class + micro-averaged) and a text report.

Predictions must already exist (run tools/dump_predictions.py first).

Usage — named detectors (recommended):
  python tools/eval_pr.py --split test --detectors emt waymo emt_waymo --nms 0.5

Usage — raw / custom files:
  python tools/eval_pr.py \
      --gt   datasets/EMT/annotations/detections_new/test_3class.json \
      --pred results/emt_test_preds.json results/waymo_test_preds.json \
      --name "COCO → EMT" "Waymo → EMT" --nms 0.5

Outputs saved to results/:
  pr_curve_{split}_nms{nms}_iou{iou}.png   — combined PR figure (2-row grid)
  pr_curve_{split}_nms{nms}_iou{iou}.txt   — AP@IoU / AP@0.5:0.95 / mAP table
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY = {
    "emt":       "results/emt_{split}_preds.json",
    "waymo":     "results/waymo_{split}_preds.json",
    "emt_waymo": "results/emt_waymo_{split}_preds.json",
}

# Paper-standard names: "source pretrain → target dataset"
DETECTOR_DISPLAY = {
    "emt":       "COCO → EMT",
    "waymo":     "COCO → Waymo",
    "emt_waymo": "Waymo → EMT",
}

GT_PATHS = {
    "test":  "datasets/EMT/annotations/detections_new/test_3class.json",
    "train": "datasets/EMT/annotations/detections_new/train_3class.json",
}

ALL_SPLITS = ("train", "test")

# Wong (2011) colorblind-safe palette + distinct linestyles for B&W printing
STYLES = [
    {"color": "#0072B2", "linestyle": "-",  "linewidth": 2.2},
    {"color": "#D55E00", "linestyle": "--", "linewidth": 2.2},
    {"color": "#009E73", "linestyle": "-.", "linewidth": 2.2},
    {"color": "#CC79A7", "linestyle": ":",  "linewidth": 2.4},
    {"color": "#E69F00", "linestyle": "-",  "linewidth": 2.2},
]

# Matplotlib settings for publication-quality figures
PAPER_RC = {
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9.5,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.8",
    "lines.linewidth":    2.2,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def iou_1_to_N(pred_box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.shape[0] == 0:
        return np.zeros(0)
    ix1 = np.maximum(pred_box[0], gt_boxes[:, 0])
    iy1 = np.maximum(pred_box[1], gt_boxes[:, 1])
    ix2 = np.minimum(pred_box[2], gt_boxes[:, 2])
    iy2 = np.minimum(pred_box[3], gt_boxes[:, 3])
    inter     = np.maximum(0., ix2 - ix1) * np.maximum(0., iy2 - iy1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area   = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union     = pred_area + gt_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gt(gt_path: str):
    with open(gt_path) as f:
        data = json.load(f)
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    raw = defaultdict(lambda: defaultdict(list))
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


def load_preds_flat(pred_path: str):
    """Returns {cat_id: [(score, image_id, box_xyxy), ...]}."""
    with open(pred_path) as f:
        preds = json.load(f)
    index = defaultdict(list)
    for det in preds:
        box = xywh_to_xyxy(np.array([det["bbox"]], dtype=np.float32))[0]
        index[det["category_id"]].append((det["score"], det["image_id"], box))
    return index


# COCO IoU thresholds: 0.50, 0.55, …, 0.95  (10 values)
COCO_IOU_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)


# ---------------------------------------------------------------------------
# PR curve computation
# ---------------------------------------------------------------------------

def compute_pr_curve(gt_index_cat, preds_cat, iou_thr, interp="101"):
    """
    Per-class PR curve via greedy highest-score-first matching.

    Returns
    -------
    prec  : (K,) array  — precision at each operating point
    rec   : (K,) array  — recall    at each operating point
    ap    : float       — area under PR curve
    n_gt  : int         — total GT boxes for this class
    scores: (N,) array  — confidence scores (for micro-averaging)
    tp    : (N,) bool   — True = TP at that score level
    """
    n_gt = sum(len(b) for b in gt_index_cat.values())

    if n_gt == 0:
        return np.array([1., 0.]), np.array([0., 0.]), 0., 0, np.array([]), np.array([], bool)
    if not preds_cat:
        return np.array([1., 0.]), np.array([0., 0.]), 0., n_gt, np.array([]), np.array([], bool)

    preds_sorted = sorted(preds_cat, key=lambda x: -x[0])
    gt_matched   = {img_id: np.zeros(len(boxes), dtype=bool)
                    for img_id, boxes in gt_index_cat.items()}

    scores = np.array([p[0] for p in preds_sorted])
    tp     = np.zeros(len(preds_sorted), dtype=bool)

    for i, (_, img_id, pred_box) in enumerate(preds_sorted):
        gt_boxes = gt_index_cat.get(img_id, np.empty((0, 4), dtype=np.float32))
        if gt_boxes.shape[0] == 0:
            continue
        iou    = iou_1_to_N(pred_box, gt_boxes)
        best_g = int(np.argmax(iou))
        m      = gt_matched.get(img_id)
        if iou[best_g] >= iou_thr and m is not None and not m[best_g]:
            m[best_g] = True
            tp[i]     = True

    tp_i  = tp.astype(int)
    fp_i  = 1 - tp_i
    tp_cum = np.cumsum(tp_i)
    fp_cum = np.cumsum(fp_i)

    rec  = tp_cum / n_gt
    prec = tp_cum / (tp_cum + fp_cum)

    rec_full  = np.concatenate([[0.], rec,  [rec[-1]]])
    prec_full = np.concatenate([[1.], prec, [0.]])
    ap        = _compute_ap(rec_full.copy(), prec_full.copy(), interp)

    return prec_full, rec_full, ap, n_gt, scores, tp


def compute_micro_pr_curve(class_raw, interp="101"):
    """
    Micro-averaged PR curve across all classes.

    class_raw : [(scores, tp_bool, n_gt), ...] one tuple per class

    Returns prec_full, rec_full, ap, sorted_scores
    (sorted_scores aligns with prec_full[1:-1] / rec_full[1:-1])
    """
    all_scores = np.concatenate([r[0] for r in class_raw if len(r[0])])
    all_tp     = np.concatenate([r[1] for r in class_raw if len(r[0])])
    total_n_gt = sum(r[2] for r in class_raw)

    if len(all_scores) == 0 or total_n_gt == 0:
        return np.array([1., 0.]), np.array([0., 0.]), 0., np.array([])

    order  = np.argsort(-all_scores)
    scores = all_scores[order]
    tp_i   = all_tp[order].astype(int)
    fp_i   = 1 - tp_i
    tp_cum = np.cumsum(tp_i)
    fp_cum = np.cumsum(fp_i)

    rec  = tp_cum / total_n_gt
    prec = tp_cum / (tp_cum + fp_cum)

    rec_full  = np.concatenate([[0.], rec,  [rec[-1]]])
    prec_full = np.concatenate([[1.], prec, [0.]])
    ap        = _compute_ap(rec_full.copy(), prec_full.copy(), interp)

    return prec_full, rec_full, ap, scores


def best_f1_point(prec_full, rec_full, scores):
    """
    Return the confidence threshold that maximises F1 on the PR curve.

    prec_full / rec_full include the leading [1,0] and trailing sentinels;
    scores aligns with the interior points prec_full[1:-1].

    Returns (threshold, precision, recall, f1).
    """
    prec_op = prec_full[1:-1]
    rec_op  = rec_full[1:-1]
    if len(prec_op) == 0 or len(scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
    denom = prec_op + rec_op
    f1    = np.where(denom > 0, 2 * prec_op * rec_op / denom, 0.0)
    i     = int(np.argmax(f1))
    return float(scores[i]), float(prec_op[i]), float(rec_op[i]), float(f1[i])


def _compute_ap(rec, prec, interp="coco"):
    """
    Area under the precision-recall curve.

    rec / prec include the leading [1, 0] and trailing sentinels added by
    compute_pr_curve; the raw operating-point values are rec[1:-1] / prec[1:-1].

    interp="coco" — Exact copy of pycocotools COCOeval.accumulate() per-curve AP:
                    monotone envelope then 101-point searchsorted lookup.
                    Use this to match eval.py / pycocotools numbers.
    interp="101"  — Our 101-point implementation (prec[rec>=thr].max()).
                    Equivalent to "coco" but uses a different code path.
    interp="all"  — All-points / Pascal VOC style area integration.
                    Slightly lower than 101-point for typical curves.
    """
    if interp == "coco":
        # ── copied verbatim from pycocotools/cocoeval.py accumulate() ──────
        # Strip sentinels; work with raw rc / pr arrays (length = N detections)
        rc = rec[1:-1]
        pr = prec[1:-1].tolist()
        nd = len(pr)
        # Smooth precision to be non-increasing (monotone envelope)
        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        # 101 recall thresholds: searchsorted → lookup precision
        rec_thrs = np.linspace(0, 1, 101)
        inds = np.searchsorted(rc, rec_thrs, side='left')
        q = np.zeros(101)
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
        except IndexError:
            pass
        return float(np.mean(q))

    if interp == "101":
        ap = 0.0
        for thr in np.linspace(0, 1, 101):
            mask = rec >= thr
            ap  += float(prec[mask].max()) if mask.any() else 0.0
        return ap / 101

    # all-points (Pascal VOC)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0] + 1
    return float(np.sum((rec[idx] - rec[idx - 1]) * prec[idx]))


def compute_coco_ap(gt_index_cat, preds_cat, interp="101"):
    """AP@0.5:0.95 — average AP over COCO IoU thresholds (mirrors COCOeval)."""
    aps = [
        compute_pr_curve(gt_index_cat, preds_cat, iou_thr, interp)[2]
        for iou_thr in COCO_IOU_THRESHOLDS
    ]
    return float(np.mean(aps))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_iso_f1(ax, levels=(0.2, 0.4, 0.6, 0.8)):
    """Draw light iso-F1 contour lines in the background."""
    x = np.linspace(0.01, 1.0, 300)
    for f in levels:
        y = f * x / (2 * x - f)
        mask = (y >= 0) & (y <= 1.05)
        ax.plot(x[mask], y[mask], color="grey", alpha=0.18,
                linewidth=0.9, linestyle=":", zorder=0)
        # label near upper-right
        xpos = min(0.88, x[mask][-1])
        ypos = f * xpos / (2 * xpos - f)
        if 0 < ypos <= 1.0:
            ax.annotate(f"F₁={f}", xy=(xpos, ypos),
                        fontsize=7, color="grey", alpha=0.6,
                        ha="left", va="bottom")


def _style_ax(ax, title, xlabel=True, ylabel=True):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel("Recall")
    if ylabel:
        ax.set_ylabel("Precision")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


# ---------------------------------------------------------------------------
# Per-class figure: one subplot per class in a single row
# ---------------------------------------------------------------------------

def plot_combined(results, categories, nms, iou, split, out_path):
    cat_ids = sorted(categories)

    # Per-class panels only — micro is in the report, not the figure
    panels = [(cat_id, categories[cat_id], "class") for cat_id in cat_ids]

    n_panels = len(panels)
    ncols    = n_panels   # one row, one column per class
    nrows    = 1

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 4.8),
                                 squeeze=False)

        legend_handles = []

        for idx, (cat_id, panel_title, panel_type) in enumerate(panels):
            ax = axes[0][idx]
            _add_iso_f1(ax)

            ap_lines = []   # (ap_value, color) for the in-plot annotation

            for det_i, res in enumerate(results):
                st    = STYLES[det_i % len(STYLES)]
                curve = res["curves"].get(cat_id)
                if curve is None:
                    continue
                prec, rec, ap = curve["prec"], curve["rec"], curve["ap"]

                ax.plot(rec, prec, **st)
                ap_lines.append((res["name"], ap, st["color"]))

                if idx == 0:
                    legend_handles.append(
                        ax.plot([], [], label=res["name"], **st)[0]
                    )

            # AP annotation block — color-coded per detector, lower-left corner
            if ap_lines:
                n     = len(ap_lines)
                row_h = 0.062          # vertical step per detector line
                pad   = 0.018
                box_y = 0.015
                box_h = row_h * n + pad * 2
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.02, box_y), 0.21, box_h,
                    transform=ax.transAxes, zorder=2,
                    boxstyle="round,pad=0.01",
                    fc="white", ec="0.75", alpha=0.88,
                ))
                for line_i, (_, ap, color) in enumerate(ap_lines):
                    ax.text(0.055, box_y + pad + line_i * row_h,
                            f"AP = {ap:.3f}",
                            transform=ax.transAxes,
                            ha="left", va="bottom", fontsize=8.5,
                            color=color, zorder=3)

            _style_ax(ax, panel_title,
                      xlabel=True,
                      ylabel=(idx == 0))

        # Single shared legend above all subplots
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(results),
            framealpha=0.9,
            edgecolor="0.8",
            fontsize=9.5,
        )
        fig.suptitle(
            f"Precision–Recall Curves  "
            f"[Split: {split} | NMS = {nms} | IoU = {iou}]",
            fontsize=13, fontweight="bold", y=1.05,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Combined PR plot → {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def build_report(results, categories, nms, iou, interp="101"):
    cat_ids    = sorted(categories)
    ap_col     = f"AP@{iou}"
    interp_lbl = {"coco": "COCO exact (pycocotools)", "101": "101-point", "all": "all-points (Pascal VOC)"}.get(interp, interp)
    lines      = [
        f"PR Curve Report  |  NMS = {nms}  IoU = {iou}  AP interpolation = {interp_lbl}",
        "=" * 80,
    ]
    for res in results:
        lines.append(f"\nDetector: {res['name']}")
        lines.append("─" * 60)

        # ── AP table ────────────────────────────────────────────────────────
        rows     = []
        aps      = []
        aps_coco = []
        for cat_id in cat_ids:
            c = res["curves"].get(cat_id)
            if c is None:
                rows.append([categories[cat_id], "—", "—", "—"])
            else:
                rows.append([categories[cat_id],
                             f"{c['ap']:.4f}", f"{c['ap_coco']:.4f}", c["n_gt"]])
                aps.append(c["ap"])
                aps_coco.append(c["ap_coco"])
        mc = res.get("micro")
        rows.append(["Micro (all classes)",
                     f"{mc['ap']:.4f}" if mc else "—", "—", ""])
        rows.append(["mAP (mean per-class)",
                     f"{np.mean(aps):.4f}"      if aps      else "—",
                     f"{np.mean(aps_coco):.4f}" if aps_coco else "—", ""])
        lines.append(tabulate(rows,
                              headers=["Class", ap_col, "AP@0.5:0.95", "N_GT"],
                              tablefmt="pipe", numalign="right"))

        # ── Best F1 threshold table ──────────────────────────────────────────
        lines.append(f"\n  Best confidence threshold (maximises F1 @ IoU {iou}):")
        thr_rows = []
        for cat_id in cat_ids:
            c = res["curves"].get(cat_id)
            if c is None:
                thr_rows.append([categories[cat_id], "—", "—", "—", "—"])
            else:
                thr_rows.append([
                    categories[cat_id],
                    f"{c['best_thr']:.4f}",
                    f"{c['best_prec']:.4f}",
                    f"{c['best_rec']:.4f}",
                    f"{c['best_f1']:.4f}",
                ])
        if mc:
            thr_rows.append([
                "Micro (all classes)",
                f"{mc['best_thr']:.4f}",
                f"{mc['best_prec']:.4f}",
                f"{mc['best_rec']:.4f}",
                f"{mc['best_f1']:.4f}",
            ])
        lines.append(tabulate(thr_rows,
                              headers=["Class", "Threshold↑", "Precision↑",
                                       "Recall↑", "F1↑"],
                              tablefmt="pipe", numalign="right"))

    if len(results) > 1:
        lines += [f"\n{'═'*80}", "Comparison", "─"*80]
        for metric_label, key in [(ap_col, "ap"), ("AP@0.5:0.95", "ap_coco")]:
            lines.append(f"\n{metric_label}")
            header = ["Class"] + [r["name"] for r in results]
            rows   = []
            for cat_id in cat_ids:
                row = [categories[cat_id]]
                for res in results:
                    c = res["curves"].get(cat_id)
                    row.append(f"{c[key]:.4f}" if c else "—")
                rows.append(row)
            micro_row = ["Micro (all)"]
            map_row   = ["mAP (mean)"]
            for res in results:
                mc   = res.get("micro")
                vals = [res["curves"][c][key] for c in cat_ids if c in res["curves"]]
                micro_row.append(f"{mc['ap']:.4f}" if (key == "ap" and mc) else "—")
                map_row.append(f"{np.mean(vals):.4f}" if vals else "—")
            rows += [micro_row, map_row]
            lines.append(tabulate(rows, headers=header, tablefmt="pipe", numalign="right"))

        # ── Best F1 threshold comparison ─────────────────────────────────────
        lines.append(f"\nBest F1 Threshold")
        header = ["Class"] + [r["name"] for r in results]
        for metric_label, thr_key in [
            ("Threshold", "best_thr"), ("F1", "best_f1"),
            ("Precision", "best_prec"), ("Recall", "best_rec"),
        ]:
            lines.append(f"\n  {metric_label}")
            rows = []
            for cat_id in cat_ids:
                row = [categories[cat_id]]
                for res in results:
                    c = res["curves"].get(cat_id)
                    row.append(f"{c[thr_key]:.4f}" if c else "—")
                rows.append(row)
            micro_row = ["Micro (all)"]
            for res in results:
                mc = res.get("micro")
                micro_row.append(f"{mc[thr_key]:.4f}" if mc else "—")
            rows.append(micro_row)
            lines.append(tabulate(rows, headers=header, tablefmt="pipe", numalign="right"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _compute_offsets(gt_paths):
    offsets = [0]
    cumulative = 0
    for gt_path in gt_paths[:-1]:
        with open(gt_path) as f:
            data = json.load(f)
        max_id = max((img["id"] for img in data["images"]), default=0)
        cumulative += max_id + 1
        offsets.append(cumulative)
    return offsets


def _merge_gt(gt_paths, offsets):
    merged = {}
    categories = None
    for gt_path, offset in zip(gt_paths, offsets):
        gt_index, cats = load_gt(gt_path)
        if categories is None:
            categories = cats
        for img_id, cats_dict in gt_index.items():
            merged[img_id + offset] = cats_dict
    return merged, categories


def load_preds_flat_multi(pred_paths, offsets):
    merged = defaultdict(list)
    for pred_path, offset in zip(pred_paths, offsets):
        if not Path(pred_path).exists():
            continue
        for cat_id, entries in load_preds_flat(pred_path).items():
            for score, img_id, box in entries:
                merged[cat_id].append((score, img_id + offset, box))
    return merged


def apply_max_dets(preds_by_cat, max_dets):
    """
    Enforce a per-image detection cap across all categories combined,
    matching pycocotools maxDets behaviour.

    Collects every prediction from every class, keeps the top max_dets
    per image by score, then rebuilds the per-category dict.
    """
    if max_dets <= 0:
        return preds_by_cat

    # Pool all predictions: (score, img_id, cat_id, box)
    all_preds = [
        (score, img_id, cat_id, box)
        for cat_id, entries in preds_by_cat.items()
        for score, img_id, box in entries
    ]

    # Group by image, keep top max_dets per image
    by_image = defaultdict(list)
    for entry in all_preds:
        by_image[entry[1]].append(entry)

    kept = defaultdict(list)
    for img_id, entries in by_image.items():
        entries.sort(key=lambda x: -x[0])
        for score, _, cat_id, box in entries[:max_dets]:
            kept[cat_id].append((score, img_id, box))

    return kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(
        description="Plot PR curves for one or more detectors (paper quality)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    named = parser.add_argument_group("Named-detector mode")
    named.add_argument("--split", default="test",
                       choices=list(GT_PATHS) + ["all"],
                       help="Split: train | test | all  (default: test).")
    named.add_argument("--detectors", nargs="+", metavar="NAME",
                       choices=list(DETECTOR_REGISTRY),
                       help=f"Named detector(s): {list(DETECTOR_REGISTRY)}")
    raw = parser.add_argument_group("Raw-file mode")
    raw.add_argument("--gt",   help="COCO ground-truth JSON.")
    raw.add_argument("--pred", nargs="+")
    raw.add_argument("--name", nargs="*")
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--nms",     type=float, default=0.5,
                        help="NMS used at dump time (for output filename only).")
    parser.add_argument("--interp",  default="coco", choices=["coco", "101", "all"],
                        help="AP interpolation: 'coco' (exact pycocotools, default) | '101' (our 101-pt) | 'all' (Pascal VOC).")
    parser.add_argument("--max-dets", type=int, default=100, metavar="N",
                        help="Max detections kept per image across all classes before evaluation "
                             "(matches pycocotools maxDets=100). Set 0 to disable.")
    parser.add_argument("--out-dir", default="results_latest")
    return parser


def main():
    args    = make_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interp_tag = {"coco": "coco", "101": "101pt", "all": "allpt"}[args.interp]
    stem       = str(out_dir / f"pr_curve_{args.split}_nms{args.nms}_iou{args.iou}_{interp_tag}")

    # ── resolve jobs ────────────────────────────────────────────────────────
    if args.detectors:
        split = args.split
        if split == "all":
            gt_paths = [GT_PATHS[s] for s in ALL_SPLITS]
            offsets  = _compute_offsets(gt_paths)
            jobs = [(
                [DETECTOR_REGISTRY[k].format(split=s) for s in ALL_SPLITS],
                offsets, gt_paths,
                DETECTOR_DISPLAY.get(k, k),
            ) for k in args.detectors]
        else:
            gt_paths = [GT_PATHS[split]]
            offsets  = [0]
            jobs = [(
                [DETECTOR_REGISTRY[k].format(split=split)],
                offsets, gt_paths,
                DETECTOR_DISPLAY.get(k, k),
            ) for k in args.detectors]

    elif args.gt and args.pred:
        gt_paths = [args.gt]
        offsets  = [0]
        names    = args.name if args.name else [Path(p).stem for p in args.pred]
        jobs     = [([p], offsets, gt_paths, n) for p, n in zip(args.pred, names)]
        split    = "custom"
    else:
        make_parser().error(
            "Provide --detectors or both --gt and --pred."
        )

    gt_index, categories = _merge_gt(jobs[0][2], jobs[0][1])
    print(f"\nSplit: {split}  |  GT images: {len(gt_index)}"
          f"  |  classes: {list(categories.values())}"
          f"\nIoU: {args.iou}  |  NMS: {args.nms}\n")

    # ── compute curves ───────────────────────────────────────────────────────
    all_results = []
    for pred_paths, offsets_j, gt_paths_j, det_name in jobs:
        missing = [p for p in pred_paths if not Path(p).exists()]
        if missing:
            for p in missing:
                print(f"  [skip] {p} not found.")
            continue

        preds_flat = load_preds_flat_multi(pred_paths, offsets_j)
        preds_flat = apply_max_dets(preds_flat, args.max_dets)
        curves     = {}
        class_raw  = []

        for cat_id in sorted(categories):
            gt_cat = {img_id: cats[cat_id]
                      for img_id, cats in gt_index.items()
                      if cat_id in cats}
            preds_cat = preds_flat.get(cat_id, [])
            prec, rec, ap, n_gt, scores, tp = compute_pr_curve(
                gt_cat, preds_cat, args.iou, args.interp
            )
            ap_coco = compute_coco_ap(gt_cat, preds_cat, args.interp)
            thr, bp, br, bf = best_f1_point(prec, rec, scores)
            curves[cat_id] = {
                "prec": prec, "rec": rec, "ap": ap,
                "ap_coco": ap_coco, "n_gt": n_gt,
                "best_thr": thr, "best_prec": bp,
                "best_rec": br,  "best_f1": bf,
            }
            class_raw.append((scores, tp, n_gt))
            print(f"  [{det_name:18s}] {categories[cat_id]:22s}  "
                  f"AP@{args.iou} = {ap:.4f}  AP@0.5:0.95 = {ap_coco:.4f}  "
                  f"best_thr = {thr:.3f}  best_F1 = {bf:.4f}  n_gt = {n_gt}")

        # micro-averaged
        m_prec, m_rec, m_ap, m_scores = compute_micro_pr_curve(class_raw, args.interp)
        m_thr, m_bp, m_br, m_bf = best_f1_point(m_prec, m_rec, m_scores)
        aps      = [curves[c]["ap"]      for c in sorted(categories)]
        aps_coco = [curves[c]["ap_coco"] for c in sorted(categories)]
        print(f"  [{det_name:18s}] {'Micro (all classes)':22s}  AP@{args.iou} = {m_ap:.4f}  "
              f"best_thr = {m_thr:.3f}  best_F1 = {m_bf:.4f}")
        print(f"  [{det_name:18s}] {'mAP (mean)':22s}  "
              f"AP@{args.iou} = {np.mean(aps):.4f}  AP@0.5:0.95 = {np.mean(aps_coco):.4f}\n")

        all_results.append({
            "name":   det_name,
            "curves": curves,
            "micro":  {
                "prec": m_prec, "rec": m_rec, "ap": m_ap,
                "best_thr": m_thr, "best_prec": m_bp,
                "best_rec": m_br,  "best_f1": m_bf,
            },
        })

    if not all_results:
        print("No results — check prediction files.")
        return

    # ── plots ────────────────────────────────────────────────────────────────
    plot_combined(all_results, categories, args.nms, args.iou, split,
                  f"{stem}.png")

    # ── text report ──────────────────────────────────────────────────────────
    report = build_report(all_results, categories, args.nms, args.iou, args.interp)
    Path(f"{stem}.txt").write_text(report)
    print(f"Report        → {stem}.txt\n")
    print(report)


if __name__ == "__main__":
    main()
