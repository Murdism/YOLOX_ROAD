#!/usr/bin/env python3
"""
Run a trained YOLOX model on a dataset split and dump COCO-format predictions
to a JSON file consumable by eval_detectors.py.

Named-detector mode (recommended)
----------------------------------
python tools/dump_predictions.py --detector emt       --split test
python tools/dump_predictions.py --detector waymo     --split test
python tools/dump_predictions.py --detector emt_waymo --split test

Raw mode (custom checkpoint / exp)
------------------------------------
python tools/dump_predictions.py \
    -f exps/example/custom/yolo_emt.py \
    -c checkpoints/final/yolox_emt_coco.pth \
    --out results/my_det_test_preds.json

Output format:
  [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
# (Python adds tools/ not the project root when running tools/dump_predictions.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tqdm import tqdm

import torch

from yolox.exp import get_exp
from yolox.utils import configure_module, fuse_model, get_model_info, postprocess, xyxy2xywh


# ---------------------------------------------------------------------------
# Detector registry — mirrors eval_detectors.py
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY = {
    "emt": {
        "exp":  "exps/example/custom/yolo_emt.py",
        "ckpt": "checkpoints/final/yolox_emt_coco.pth",
        "pred": "results/emt_{split}_preds.json",
    },
    "waymo": {
        "exp":  "exps/example/custom/yolo_emt_from_waymo.py",
        "ckpt": "checkpoints/final/yolox_road_waymo.pth",
        "pred": "results/waymo_{split}_preds.json",
    },
    "emt_waymo": {
        "exp":  "exps/example/custom/yolo_emt_from_waymo.py",
        "ckpt": "checkpoints/final/yolox_emt_waymo.pth",
        "pred": "results/emt_waymo_{split}_preds.json",
    },
}

GT_ANNS = {
    "test":  "datasets/EMT/annotations/detections_new/test_3class.json",
    "train": "datasets/EMT/annotations/detections_new/train_3class.json",
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataloader, num_classes, conf_thr, nms_thr, test_size, half):
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    model = model.eval()
    if half:
        model = model.half()

    data_list = []
    for imgs, _, info_imgs, ids in tqdm(dataloader, desc="Inference"):
        with torch.no_grad():
            outputs = model(imgs.type(tensor_type))
            outputs = postprocess(outputs, num_classes, conf_thr, nms_thr)

        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]
            scale  = min(test_size[0] / float(img_h), test_size[1] / float(img_w))
            bboxes /= scale
            bboxes_xywh = xyxy2xywh(bboxes)
            cls    = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for i in range(bboxes.shape[0]):
                label = dataloader.dataset.class_ids[int(cls[i])]
                data_list.append({
                    "image_id":    int(img_id),
                    "category_id": label,
                    "bbox":        bboxes_xywh[i].numpy().tolist(),
                    "score":       float(scores[i]),
                })

    return data_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser(
        description="Dump YOLOX predictions to COCO-format JSON for eval_detectors.py",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    named = parser.add_argument_group("Named-detector mode")
    named.add_argument(
        "--detector", choices=list(DETECTOR_REGISTRY),
        help=f"Named detector: {list(DETECTOR_REGISTRY)}",
    )
    named.add_argument(
        "--split", default="test", choices=list(GT_ANNS),
        help="Which split to run inference on: train | test  (default: test).",
    )

    raw = parser.add_argument_group("Raw mode (custom paths)")
    raw.add_argument("-f", "--exp_file", help="Experiment .py file.")
    raw.add_argument("-c", "--ckpt",     help="Checkpoint .pth file.")
    raw.add_argument("--out",            help="Output JSON path.")

    parser.add_argument("--conf",  type=float, default=0.01,
                        help="Conf threshold during NMS (default: 0.01, matches experiment). "
                             "Use 0.001 only if you need the full low-confidence tail of the PR curve.")
    parser.add_argument("--nms",   type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5).")
    parser.add_argument("--tsize", type=int,   default=None,
                        help="Test image size (overrides exp.test_size).")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--fp16",  action="store_true")
    parser.add_argument("--fuse",  action="store_true")
    return parser


def main():
    configure_module()
    args = make_parser().parse_args()

    # ── resolve exp / ckpt / out ────────────────────────────────────────────
    if args.detector:
        reg      = DETECTOR_REGISTRY[args.detector]
        exp_file = reg["exp"]
        ckpt     = reg["ckpt"]
        out      = reg["pred"].format(split=args.split)
    elif args.exp_file and args.ckpt and args.out:
        exp_file = args.exp_file
        ckpt     = args.ckpt
        out      = args.out
    else:
        make_parser().error(
            "Provide --detector (named mode) or all of -f / -c / --out (raw mode)."
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exp = get_exp(exp_file, None)
    if args.conf  is not None: exp.test_conf = args.conf
    if args.nms   is not None: exp.nmsthre   = args.nms
    if args.tsize is not None: exp.test_size  = (args.tsize, args.tsize)

    # Override val annotation to the requested split
    if args.detector:
        exp.val_ann  = Path(GT_ANNS[args.split]).name
        exp.test_ann = exp.val_ann

    logger.info(f"Exp       : {exp_file}")
    logger.info(f"Checkpoint: {ckpt}")
    logger.info(f"Output    : {out}")
    logger.info(f"conf={exp.test_conf}  nms={exp.nmsthre}  size={exp.test_size}")

    model = exp.get_model()
    logger.info(get_model_info(model, exp.test_size))

    if args.device == "gpu":
        model.cuda()

    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model"])
    logger.info("Checkpoint loaded.")

    if args.fuse:
        model = fuse_model(model)

    val_loader = exp.get_evaluator(
        batch_size=args.batch_size,
        is_distributed=False,
        testdev=False,
    ).dataloader

    data_list = run_inference(
        model, val_loader,
        num_classes=exp.num_classes,
        conf_thr=exp.test_conf,
        nms_thr=exp.nmsthre,
        test_size=exp.test_size,
        half=args.fp16 and args.device == "gpu",
    )

    out_path.write_text(json.dumps(data_list))
    logger.info(f"Saved {len(data_list)} detections → {out}")


if __name__ == "__main__":
    main()
