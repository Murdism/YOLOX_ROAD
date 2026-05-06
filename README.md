<div align="center"><img src="assets/logo.png" width="300"></div>

# YOLOX-ROAD

A fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for road-scene object detection on the **EMT** and **ROAD Waymo** datasets.

This fork preserves the full upstream YOLOX training and deployment stack and adds:

- **EMT dataset support** — 3-class road-scene taxonomy (VRU, Two-Wheeler, Vehicle)
- **ROAD Waymo dataset support** — converted to COCO JSON for standard YOLOX training
- **Rare-class oversampling** — image-level weighted sampler ([`WeightedInfiniteSampler`](yolox/data/samplers.py))
- **Per-class weighted loss** — configurable per-class BCE weights in the detection head ([`yolo_head.py`](yolox/models/yolo_head.py))
- **Dataset conversion utilities** — KITTI-style and ROAD Waymo → COCO JSON converters
- **Annotation visualization tools** — KITTI TXT and COCO JSON label rendering

---

## Sample Detections

| EMT | ROAD Waymo |
|:---:|:---:|
| <img src="visualizations/emt/video_1_000001.jpg" width="460"> | <img src="visualizations/road_waymo/train_00052/00001.jpg" width="460"> |

Classes: **VulnerableRoadUser** (pedestrian/cyclist) · **Two-Wheeler** (motorbike/small motorised) · **Vehicle** (car/bus/van/emergency)

---

## Documentation

| Document | Description |
|---|---|
| [datasets/README.md](datasets/README.md) | Dataset directory layout, conversion commands, class taxonomy |
| [tools/support_scripts/README.md](tools/support_scripts/README.md) | All conversion, inspection, and dataset management scripts |

---

## Detector Checkpoints

Checkpoints are too large to commit (~430 MB each). Download from Google Drive:

| File | Link | mAP |
|---|---|---|
| `yolox_emt_coco.pth` | [Drive](https://drive.google.com/file/d/14lHFGtxg0GQtkLjNLLzHMHAGwTTkGoIH/view?usp=sharing) | 0.287 (EMT) |
| `yolox_emt_waymo.pth` | [Drive](https://drive.google.com/file/d/16_YmZiwy58sD1SKIY5b4F-hpqCCnhQCL/view?usp=sharing) | 0.287 (EMT) |
| `yolox_road_waymo.pth` | [Drive](https://drive.google.com/file/d/1EZs1VSeQECoOWuASN6Pbt3b0pX70MOkl/view?usp=drive_link) | 0.307 (Waymo) |

Place files in `checkpoints/final/`.

### Per-class performance on EMT test set

AP@0.5:0.95, fp32, 1280×1280 input.

| Checkpoint | VRU | Two-Wheeler | Vehicle | mAP |
|---|---|---|---|---|
| `yolox_road_waymo.pth` (cross-domain) | 17.1 | 2.9 | 28.6 | 0.162 |
| `yolox_emt_waymo.pth` | 21.5 | 21.6 | 43.1 | 0.287 |
| `yolox_emt_coco.pth` | 21.4 | 22.0 | 42.7 | 0.287 |

---

## Fork-Specific Additions

### Rare-Class Oversampling

Imbalanced road datasets under-represent VRUs and two-wheelers. This fork adds [`WeightedInfiniteSampler`](yolox/data/samplers.py) which upsamples images containing rare classes at training time.

**How it works:**

```
repeat_factor(image) = max(
    max_image_frequency / class_frequency(c)
    for c in image if c is a target class
)
```

The sampler draws images via `torch.multinomial` with replacement, so rare-class images appear proportionally more often without duplicating data on disk. Distributed training is fully supported.

**Config knobs** (set in the exp file):

| Parameter | Default | Description |
|---|---|---|
| `enable_rare_class_oversampling` | `True` | Toggle the feature |
| `oversample_target_classes` | `("VulnerableRoadUser", "Two-Wheeler")` | Classes to oversample |
| `max_oversample_factor` | 4.0 (EMT) / 8.0 (Road) | Cap on per-image repeat factor |
| `auto_select_oversample_classes` | `False` | Auto-detect rare classes below threshold |
| `oversample_minority_ratio_threshold` | 0.2 | Frequency threshold for auto-selection |

Before training starts, the sampler logs per-class annotation counts and the projected distribution after weighting so you can verify the setup.

### Per-Class Weighted Loss

The detection head ([`yolox/models/yolo_head.py`](yolox/models/yolo_head.py)) accepts optional per-class weights applied to the binary cross-entropy classification loss:

```python
cls_loss_raw = BCEWithLogitsLoss()(cls_preds, cls_targets)   # [num_fg, num_classes]
if cls_loss_weights is not None:
    loss_cls = (cls_loss_raw * cls_loss_weights).sum() / num_fg
else:
    loss_cls = cls_loss_raw.sum() / num_fg
```

Default weights (order matches sorted category IDs):

| Class | ID | Weight (EMT) | Weight (Road-Waymo) |
|---|---|---|---|
| VulnerableRoadUser | 1 | 2.0 | 1.5 |
| Two-Wheeler | 2 | 3.0 | 6.0 |
| Vehicle | 3 | 1.0 | 1.0 |

Set `cls_loss_weights` in the exp file to adjust or disable.

---

## Installation

```shell
git clone https://github.com/Murdism/YOLOX_ROAD.git
cd YOLOX_ROAD
pip install -v -e .
```

Requirements: Python ≥ 3.7, PyTorch ≥ 1.7, CUDA recommended. See [requirements.txt](requirements.txt).

---

## Dataset Setup

> Full layout details and conversion commands are in [datasets/README.md](datasets/README.md).

### EMT

```
$YOLOX_DATADIR/EMT/
├── frames/                            # training images
├── test_frames/                       # test images
└── annotations/
    └── detections_new/
        ├── train_3class.json          # generated by emt_to_coco.py
        └── test_3class.json
```

Convert KITTI-style labels before the first training run:

```shell
python tools/support_scripts/emt_to_coco.py
```

### ROAD Waymo

```
$YOLOX_DATADIR/road_waymo/
├── train_frames/                      # all frames (train and val share this directory)
├── road_waymo_trainval_v1.0.json      # original source annotations
└── road_waymo_annotations/
    ├── train_3class.json              # generated by road_waymo_tool.py merge
    └── val_3class.json
```

Generate the 3-class annotation files:

```shell
python tools/support_scripts/road_waymo_tool.py merge --split train --preset emt \
    --out $YOLOX_DATADIR/road_waymo/road_waymo_annotations/train_3class.json

python tools/support_scripts/road_waymo_tool.py merge --split val --preset emt \
    --out $YOLOX_DATADIR/road_waymo/road_waymo_annotations/val_3class.json
```

For more annotation inspection and remapping options see [tools/support_scripts/README.md](tools/support_scripts/README.md).

---

## Training

### EMT — from COCO pretrained weights

```shell
python -m yolox.tools.train \
  -f exps/example/custom/yolo_emt.py \
  -d 1 -b 8 --fp16 -o \
  -c pretrained/yolox_l.pth
```

### EMT — fine-tune from ROAD-Waymo checkpoint

```shell
python -m yolox.tools.train \
  -f exps/example/custom/yolo_emt_from_waymo.py \
  -d 1 -b 8 --fp16 -o \
  -c checkpoints/final/yolox_road_waymo.pth
```

### ROAD Waymo

```shell
python -m yolox.tools.train \
  -f exps/example/custom/yolo_road_waymo.py \
  -d 1 -b 8 --fp16 -o \
  -c pretrained/yolox_l.pth
```

Common flags: `-d` = number of GPUs, `-b` = total batch size, `--cache` = cache images in RAM.

Checkpoints are saved under `checkpoints/<exp_name>/` (`yolox_emt`, `yolox_emt_from_waymo`, or `yolox_road_waymo`).

### Training configuration reference

| Parameter | EMT (COCO) | EMT (Waymo) | ROAD-Waymo |
|---|---|---|---|
| Exp file | [`yolo_emt.py`](exps/example/custom/yolo_emt.py) | [`yolo_emt_from_waymo.py`](exps/example/custom/yolo_emt_from_waymo.py) | [`yolo_road_waymo.py`](exps/example/custom/yolo_road_waymo.py) |
| Input size | 1280×1280 | 1280×1280 | 1280×1280 |
| Max epochs | 120 | 30 | 60 |
| Warmup epochs | 5 | 1 | 3 |
| LR per image | 0.001/64 | 0.0001/64 | 0.001/64 |
| Class weights (VRU / TW / Veh) | 2.0 / 3.0 / 1.0 | 1.5 / 2.0 / 1.0 | 1.5 / 6.0 / 1.0 |
| Oversample factor | 4.0 | 3.0 | 8.0 |
| Oversample targets | VRU, Two-Wheeler | VRU, Two-Wheeler | Two-Wheeler only |
| Mosaic / Mixup prob | 0.8 / 0.5 | 0.5 / 0.3 | 0.8 / 0.5 |

---

## Evaluation

Expected results match the [per-class table above](#per-class-performance-on-emt-test-set).

```shell
# yolox_emt_coco.pth on EMT  →  expect mAP 0.287
python tools/eval.py \
  -f exps/example/custom/yolo_emt.py \
  -c checkpoints/final/yolox_emt_coco.pth \
  -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# yolox_emt_waymo.pth on EMT  →  expect mAP 0.287
python tools/eval.py \
  -f exps/example/custom/yolo_emt_from_waymo.py \
  -c checkpoints/final/yolox_emt_waymo.pth \
  -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# yolox_road_waymo.pth on EMT (cross-domain)  →  expect mAP 0.162
python tools/eval.py \
  -f exps/example/custom/yolo_emt_from_waymo.py \
  -c checkpoints/final/yolox_road_waymo.pth \
  -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# yolox_road_waymo.pth on Waymo test  →  expect mAP 0.307
python tools/eval.py \
  -f exps/example/custom/yolo_road_waymo.py \
  -c checkpoints/final/yolox_road_waymo.pth \
  -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280
```

---

## Annotation Visualization

> Full usage reference is in [tools/support_scripts/README.md](tools/support_scripts/README.md).

```shell
# Single KITTI-labeled frame
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path $YOLOX_DATADIR/EMT/emt_annotations/labels_full \
  --images-root $YOLOX_DATADIR/EMT/frames \
  --mode sample --video-name video_054604 --frame-id 149

# Single COCO-labeled frame
python tools/support_scripts/visualize_annotations.py \
  --label-format coco \
  --labels-path $YOLOX_DATADIR/EMT/annotations/detections_new/train_3class.json \
  --images-root $YOLOX_DATADIR/EMT/frames \
  --mode sample --video-name video_054604 --frame-id 149

# Render and save a full video sequence
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path $YOLOX_DATADIR/EMT/emt_annotations/labels_full \
  --images-root $YOLOX_DATADIR/EMT/frames \
  --mode video --video-name video_054604 --save-video
```

---

## Class Taxonomy

Both datasets use a unified 3-class scheme. See [datasets/README.md](datasets/README.md) for source-category mappings.

| ID | Class | Source Categories |
|---|---|---|
| 1 | VulnerableRoadUser | Pedestrian, Cyclist |
| 2 | Two-Wheeler | Motorbike, Small_motorised_vehicle |
| 3 | Vehicle | Car, Bus, Medium_vehicle, Large_vehicle, Emergency_vehicle |

Mapping scripts: [`emt_to_superclass.py`](tools/support_scripts/emt_to_superclass.py) · [`road_waymo_tool.py --preset emt`](tools/support_scripts/road_waymo_tool.py)

---

## Project Layout

```
YOLOX_ROAD/
├── exps/example/custom/               # quick-start experiment wrappers
│   ├── yolo_emt.py                    # EMT from COCO pretrained
│   ├── yolo_emt_from_waymo.py         # EMT fine-tuned from Waymo checkpoint
│   └── yolo_road_waymo.py             # ROAD Waymo
├── yolox/
│   ├── data/
│   │   ├── datasets/
│   │   │   └── emt_dataset.py         # EMTDataset (shared by both EMT exp files)
│   │   └── samplers.py                # WeightedInfiniteSampler
│   ├── exp/
│   │   ├── yolox_emt.py               # EMT experiment definition
│   │   ├── yolox_emt_from_waymo.py    # fine-tuning variant
│   │   ├── yolox_road.py              # ROAD Waymo experiment + RoadDataset
│   │   └── yolox_road_uk.py           # ROAD UK experiment
│   └── models/
│       └── yolo_head.py               # per-class weighted BCE loss
├── tools/
│   └── support_scripts/               # see tools/support_scripts/README.md
│       ├── emt_to_coco.py             # KITTI → COCO conversion
│       ├── road_to_coco.py            # ROAD Waymo → COCO conversion
│       ├── road_waymo_tool.py         # stats, visualization, class remapping
│       └── visualize_annotations.py  # KITTI / COCO label renderer
├── datasets/                          # README.md describes expected layout
└── checkpoints/final/                 # place downloaded .pth files here
```

---

## Original YOLOX Reference

This fork is built on top of the official **[Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** repository. The upstream README covers standard COCO benchmarks, generic installation and demo usage, multi-GPU training, and ONNX / TensorRT / ncnn / OpenVINO export.

<details>
<summary>Upstream benchmark (COCO val)</summary>

| Model | Size | mAP val 0.5:0.95 | Params (M) | FLOPs (G) | Weights |
|---|---|---|---|---|---|
| YOLOX-s | 640 | 40.5 | 9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
| YOLOX-m | 640 | 46.9 | 25.3 | 73.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
| YOLOX-l | 640 | 49.7 | 54.2 | 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| YOLOX-x | 640 | 51.1 | 99.1 | 281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |

</details>

---

## Citation

If you use this work, please cite the EMT dataset paper:

```bibtex
@misc{madjid2025emtvisualmultitaskbenchmark,
  title         = {EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving},
  author        = {Nadya Abdel Madjid and Murad Mebrahtu and Abdulrahman Ahmad and
                   Abdelmoamen Nasser and Bilal Hassan and Naoufel Werghi and
                   Jorge Dias and Majid Khonji},
  year          = {2025},
  eprint        = {2502.19260},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2502.19260},
}
```

If you use the underlying detector, please also cite the original YOLOX paper:

```bibtex
@article{yolox2021,
  title   = {YOLOX: Exceeding YOLO Series in 2021},
  author  = {Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal = {arXiv preprint arXiv:2107.08430},
  year    = {2021}
}
```
