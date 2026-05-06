<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).

This repo is an implementation of PyTorch version YOLOX, there is also a [MegEngine implementation](https://github.com/MegEngine/YOLOX).

<img src="assets/git_fig.png" width="1000" >

## About This Fork

This repository is an extension of the original [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) project. The upstream project is still the main reference for the core detector, training engine, and deployment stack.

This fork focuses on our added work on top of YOLOX, especially:
- EMT dataset support
- ROAD Waymo dataset support
- custom dataset conversion utilities
- annotation visualization and inspection helpers

The upstream README content is still useful and remains below, but the sections in this README that matter most for this fork are the fork-specific setup and workflow notes.

## Fork-Specific Additions

- Added an EMT experiment file at [yolox/exp/yolox_emt.py](./yolox/exp/yolox_emt.py) and a thin custom exp wrapper at [exps/example/custom/yolo_emt.py](./exps/example/custom/yolo_emt.py).
- Added COCO conversion utilities for custom datasets:
  - [emt_to_coco.py](./emt_to_coco.py)
  - [road_to_coco.py](./road_to_coco.py)
- Added a support-scripts folder at [tools/support_scripts](./tools/support_scripts) for reusable dataset helpers and inspection tools.
- Added an annotation visualizer at [tools/support_scripts/visualize_annotations.py](./tools/support_scripts/visualize_annotations.py) that supports both KITTI TXT labels and COCO JSON labels.
- EMT conversion currently expects KITTI-style tracking labels under `datasets/emt/emt_annotations/labels_full` and writes YOLOX-ready COCO JSON files to:
  - `datasets/emt/emt_annotations/train.json`
  - `datasets/emt/emt_annotations/test.json`

## Recommended Reading Order

If you are working in this fork, use this order:

1. Read the fork-specific sections in this README first.
2. Use the scripts under [tools/support_scripts](./tools/support_scripts) for conversion and visualization.
3. Use the EMT experiment files in:
   - [yolox/exp/yolox_emt.py](./yolox/exp/yolox_emt.py)
   - [exps/example/custom/yolo_emt.py](./exps/example/custom/yolo_emt.py)
4. Refer back to the original YOLOX sections below for generic installation, training, evaluation, and deployment behavior.

## Updates!!
* 【2023/02/28】 We support assignment visualization tool, see doc [here](./docs/assignment_visualization.md).
* 【2022/04/14】 We support jit compile op.
* 【2021/08/19】 We optimize the training process with **2x** faster training and **~1%** higher performance! See [notes](docs/updates_note.md) for more details.
* 【2021/08/05】 We release [MegEngine version YOLOX](https://github.com/MegEngine/YOLOX).
* 【2021/07/28】 We fix the fatal error of [memory leak](https://github.com/Megvii-BaseDetection/YOLOX/issues/103)
* 【2021/07/26】 We now support [MegEngine](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/MegEngine) deployment.
* 【2021/07/20】 We have released our technical report on [Arxiv](https://arxiv.org/abs/2107.08430).

## Benchmark

#### Standard Models.

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

</details>

#### Light Models.

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |


<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.3  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.pth) |

</details>

## Quick Start

### EMT Workflow

The current EMT pipeline in this fork is:

1. Keep EMT frames under:
   - `datasets/emt/frames`
   - `datasets/emt/test_frames`
2. Keep KITTI-style labels under:
   - `datasets/emt/emt_annotations/labels_full`
3. Convert KITTI labels to COCO JSON before training:

```shell
python emt_to_coco.py
```

or

```shell
python tools/support_scripts/emt_to_coco.py
```

This writes:
- `datasets/emt/emt_annotations/train.json`
- `datasets/emt/emt_annotations/test.json`

The generated COCO files are what the EMT experiment reads during training and evaluation.

To train with the EMT experiment:

```shell
python -m yolox.tools.train -f exps/example/custom/yolo_emt.py -d 1 -b 2 --fp16 -o
```

Adjust `-d` and `-b` for your machine.

To fine-tune from pretrained YOLOX weights, pass the checkpoint with `-c`. Since the EMT experiment uses `depth = 1.33` and `width = 1.25`, the matching upstream starting point is `YOLOX-X`.

```shell
  python -m yolox.tools.train \
  -f exps/example/custom/yolo_emt.py \
  -d 1 \
  -b 8 \
  --fp16 \
  -o \
  -c pretrained/yolox_l.pth

```
```shell 
python -m yolox.tools.train \
  -f exps/example/custom/yolo_road_uk.py \
  -d 1 \
  -b 12 \
  --fp16 \
  -o \
  -c pretrained/yolox_l.pth \
  --cache
```
```shell 
python -m yolox.tools.train \
  -f exps/example/custom/yolo_road.py \
  -d 1 \
  -b 12 \
  --fp16 \
  -o \
  -c pretrained/yolox_l.pth \
  --cache
```
To train the Road Waymo experiment with the custom Road config:

```shell
python -m yolox.tools.train \
  -f exps/example/custom/yolo_road_waymo.py \
  -d 1 \
  -b 2 \
  --fp16 \
  -o \
  -c pretrained/yolox_l.pth
```

Explanation of the Road Waymo training command:
- `-f exps/example/custom/yolo_road_waymo.py`: experiment description file for Road Waymo.
- `-d 1`: use one GPU/device for training.
- `-b 2`: batch size for training.
- `--fp16`: enable mixed precision (AMP) training.
- `-o`: occupy GPU memory first before training starts.
- `-c pretrained/yolox_l.pth`: initialize the model from the pretrained YOLOX-L checkpoint.

The EMT experiment now saves outputs under:

```text
checkpoints/yolox_emt/
```

Typical checkpoint files inside that folder are:
- `latest_ckpt.pth`
- `last_epoch_ckpt.pth`
- `best_ckpt.pth`
- `epoch_<n>_ckpt.pth` when history saving is enabled

# Detector Checkpoints

Checkpoint files are not committed to the repository (each is ~430MB).
This document describes the available checkpoints and how to obtain them.

## Available Checkpoints

| Checkpoint | Pretraining | Fine-tune | EMT mAP | Use Case |
|---|---|---|---|---|
| `final/yolox_emt_coco.pth` | COCO | EMT | 0.287 | Main fine-tuned detector |
| `final/yolox_emt_waymo.pth` | COCO + ROAD-Waymo | EMT | 0.287 | Cross-dataset pretraining ablation |
| `final/yolox_road_waymo.pth` | COCO | ROAD-Waymo | 0.162* | Off-the-shelf baseline / Waymo detector |

*0.162 mAP on EMT (cross-domain). On Waymo test set: 0.307 mAP.

## Per-Class Performance on EMT Test Set

All values are AP@0.5:0.95, fp32 evaluation, 1280×1280 input.

| Detector | VRU | Two-Wheeler | Vehicle | mAP |
|---|---|---|---|---|
| `yolox_road_waymo.pth` | 17.13 | 2.87 | 28.63 | 0.162 |
| `yolox_emt_waymo.pth` | 21.48 | 21.57 | 43.10 | 0.287 |
| `yolox_emt_coco.pth` | 21.35 | 21.97 | 42.72 | 0.287 |

## How to Obtain

### Option A: Download from GitHub Release

```bash
mkdir -p checkpoints/final

# Update USERNAME/REPO and TAG to match your release
BASE="https://github.com/USERNAME/REPO/releases/download/v1.0"

wget $BASE/yolox_emt_coco.pth   -O checkpoints/final/yolox_emt_coco.pth
wget $BASE/yolox_emt_waymo.pth  -O checkpoints/final/yolox_emt_waymo.pth
wget $BASE/yolox_road_waymo.pth -O checkpoints/final/yolox_road_waymo.pth
```

### Option B: Re-train locally

Each checkpoint can be reproduced from scratch using the configs in
`exps/example/custom/`. See the main `README.md` and `docs/TRAINING.md`
for full commands.

Approximate training time on a single RTX 5090:
- ROAD-Waymo: ~16 hours (24 epochs)
- EMT (from COCO): ~13 hours (60 epochs, peak at epoch 6)
- EMT (from ROAD-Waymo): ~6 hours (30 epochs, peak at epoch 6)

### Option C: Contact authors

Email [your-email@example.com] for direct access.

## Expected Directory Structure

After downloading, your `checkpoints/` directory should look like:

```
checkpoints/
├── README.md                          (this file)
└── final/
    ├── yolox_emt_coco.pth            (~430MB)
    ├── yolox_emt_waymo.pth           (~430MB)
    └── yolox_road_waymo.pth          (~430MB)
```

## Verification

After downloading, verify each checkpoint by running evaluation:

```bash
# Verify yolox_emt_coco.pth (expect mAP 0.287)
python tools/eval.py \
    -f exps/example/custom/yolo_emt.py \
    -c checkpoints/final/yolox_emt_coco.pth \
    -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# Verify yolox_emt_waymo.pth (expect mAP 0.287)
python tools/eval.py \
    -f exps/example/custom/yolo_emt_from_waymo.py \
    -c checkpoints/final/yolox_emt_waymo.pth \
    -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# Verify yolox_road_waymo.pth on Waymo test (expect mAP 0.307)
python tools/eval.py \
    -f exps/example/custom/yolo_road_waymo.py \
    -c checkpoints/final/yolox_road_waymo.pth \
    -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280

# Verify cross-domain: Waymo detector on EMT test (expect mAP 0.162)
python tools/eval.py \
    -f exps/example/custom/yolo_emt_from_waymo.py \
    -c checkpoints/final/yolox_road_waymo.pth \
    -b 8 -d 1 --conf 0.01 --nms 0.5 --tsize 1280
```

## Class Taxonomy

All checkpoints use a unified 3-class taxonomy:

| ID | Class | Source Categories |
|---|---|---|
| 1 | VulnerableRoadUser | Pedestrian, Cyclist |
| 2 | Two-Wheeler | Motorbike, Small motorised vehicle |
| 3 | Vehicle | Car, Bus, Medium/Large vehicle, Emergency vehicle |

See `tools/remap_emt_to_3class.py` for the mapping script.

## Training Configuration Summary

| Parameter | EMT (COCO) | EMT (Waymo) | ROAD-Waymo |
|---|---|---|---|
| Input size | 1280×1280 | 1280×1280 | 1280×1280 |
| Batch size | 8 | 8 | 8 |
| Initial LR | 0.001/64 | 0.0001/64 | 0.001/64 |
| Max epochs | 120 | 30 | 60 |
| Peak epoch | 6 | 6 | 4 |
| Class weights (VRU, TW, Veh) | [2.0, 3.0, 1.0] | [1.5, 2.0, 1.0] | [1.5, 6.0, 1.0] |
| Oversample factor | 4.0 | 3.0 | 8.0 |
| Mosaic / Mixup | 0.8 / 0.5 | 0.5 / 0.3 | 0.8 / 0.5 |

## File Sizes

| File | Size |
|---|---|
| `yolox_emt_coco.pth` | ~430 MB |
| `yolox_emt_waymo.pth` | ~430 MB |
| `yolox_road_waymo.pth` | ~430 MB |
| **Total** | **~1.3 GB** |
### Visualizing Labels

Visualize one KITTI-labeled sample:

```shell
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path datasets/emt/emt_annotations/labels_full \
  --images-root datasets/emt/frames \
  --mode sample \
  --video-name video_054604 \
  --frame-id 149
```

Visualize one COCO-labeled sample:

```shell
python tools/support_scripts/visualize_annotations.py \
  --label-format coco \
  --labels-path datasets/emt/emt_annotations/train.json \
  --images-root datasets/emt/frames \
  --mode sample \
  --video-name video_054604 \
  --frame-id 149
```

Render a whole video folder:

```shell
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path datasets/emt/emt_annotations/labels_full \
  --images-root datasets/emt/frames \
  --mode video \
  --video-name video_054604 \
  --save-video
```

### ROAD Waymo Workflow

Convert ROAD Waymo annotations to COCO JSON:
```shell
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo
```

Note:
- `datasets/road_waymo/road_waymo_trainval_v1.0.json` is the original ROAD Waymo source annotation file.
- This repo also includes converted COCO-style split files at `datasets/road_waymo/annotations/train.json` and `datasets/road_waymo/annotations/val.json`.
- The `yolo_road_waymo` experiment expects those COCO JSON files, not the raw ROAD source format.
- By default, the Road experiment loads images from `datasets/road_waymo/train_frames` for both training and validation, while `train.json` and `val.json` select which frames belong to each split.
- Current converted counts are:
  - train: `115602` images, `2515836` annotations
  - val: `37932` images, `788517` annotations

If you want YOLOX-style split folders created under the output dataset root, add:

```shell
python road_to_coco.py --road-dir datasets/road_waymo --prepare-yolox-layout
```

#### ROAD Waymo Annotation Tool

[tools/road_waymo_tool.py](./tools/road_waymo_tool.py) is a self-contained script for inspecting and remapping the ROAD Waymo class annotations. It reads directly from `datasets/road_waymo/road_waymo_annotations/`.

**Statistics**

```shell
python tools/road_waymo_tool.py stats --split train
python tools/road_waymo_tool.py stats --split val
```

**Visualize annotated frames**

```shell
# Random frames from train set (space/n = next, p = prev, q = quit)
python tools/road_waymo_tool.py visualize --split train

# One specific video sequence, in frame order
python tools/road_waymo_tool.py visualize --split train --video train_00444

# Auto-play at ~5 fps
python tools/road_waymo_tool.py visualize --split train --video train_00444 --delay 200
```

**Merge / remap classes**

Produce the EMT 3-class JSON (matches the category scheme used in `yolox_emt.py`):

```shell
python tools/road_waymo_tool.py merge --split train --preset emt \
    --out datasets/road_waymo/road_waymo_annotations/train_3class.json

python tools/road_waymo_tool.py merge --split val --preset emt \
    --out datasets/road_waymo/road_waymo_annotations/val_3class.json
```

The `emt` preset maps the 9 original classes to:

| id | Name | Original classes |
|----|------|-----------------|
| 1 | VulnerableRoadUser | Pedestrian, Cyclist |
| 2 | Two-Wheeler | Motorbike, Small_motorised_vehicle |
| 3 | Vehicle | Bus, Car, Large_vehicle, Medium_vehicle, Emergency_vehicle |

Custom merges are also supported with `--merge "NewName:Class1,Class2"` (repeatable). Add `--drop-unmapped` to silently discard any class not covered by a merge spec:

```shell
python tools/road_waymo_tool.py merge --split train \
    --merge "Vehicle:Bus,Car,Large_vehicle,Medium_vehicle,Emergency_vehicle" \
    --merge "VRU:Pedestrian,Cyclist" \
    --drop-unmapped \
    --out datasets/road_waymo/road_waymo_annotations/train_2class.json
```

This fork keeps the original YOLOX training and evaluation flow, but uses custom dataset conversion and experiment files for our datasets.

<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]
                               yolox-m
                               yolox-l
                               yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training
* --cache: caching imgs into RAM to accelarate training, which need large system RAM.



When using -f, the above commands are equivalent to:
```shell
python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]
                               exps/default/yolox_m.py
                               exps/default/yolox_l.py
                               exps/default/yolox_x.py
```

**Multi Machine Training**

We also support multi-nodes training. Just add the following args:
* --num\_machines: num of your total training nodes
* --machine\_rank: specify the rank of each node

Suppose you want to train YOLOX on 2 machines, and your master machines's IP is 123.123.123.123, use port 12312 and TCP.

On master machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 0
```
On the second machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 1
```

**Logging to Weights & Biases**

To log metrics, predictions and model checkpoints to [W&B](https://docs.wandb.ai/guides/integrations/other/yolox) use the command line argument `--logger wandb` and use the prefix "wandb-" to specify arguments for initializing the wandb run.

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache] --logger wandb wandb-project <project name>
                         yolox-m
                         yolox-l
                         yolox-x
```

An example wandb dashboard is available [here](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom0)

**Others**

See more information with the following command:
```shell
python -m yolox.tools.train --help
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                               yolox-m
                               yolox-l
                               yolox-x
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
                               yolox-m
                               yolox-l
                               yolox-x
```

</details>


<details>
<summary>Tutorials</summary>

*  [Training on custom data](docs/train_custom_data.md)
*  [Caching for custom data](docs/cache.md)
*  [Manipulating training image size](docs/manipulate_training_image_size.md)
*  [Assignment visualization](docs/assignment_visualization.md)
*  [Freezing model](docs/freeze_module.md)

</details>

## Deployment


1. [MegEngine in C++ and Python](./demo/MegEngine)
2. [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
3. [TensorRT in C++ and Python](./demo/TensorRT)
4. [ncnn in C++ and Java](./demo/ncnn)
5. [OpenVINO in C++ and Python](./demo/OpenVINO)
6. [Accelerate YOLOX inference with nebullvm in Python](./demo/nebullvm)

## Third-party resources
* YOLOX for streaming perception: [StreamYOLO (CVPR 2022 Oral)](https://github.com/yancie-yjr/StreamYOLO)
* The YOLOX-s and YOLOX-nano are Integrated into [ModelScope](https://www.modelscope.cn/home). Try out the Online Demo at [YOLOX-s](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox/summary) and [YOLOX-Nano](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox_nano_coco/summary) respectively 🚀.
* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sultannn/YOLOX-Demo)
* The ncnn android app with video support: [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* YOLOX with Tengine support: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX Deploy DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX MNN/TNN/ONNXRuntime: [YOLOX-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolox.cpp)、[YOLOX-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolox.cpp) and [YOLOX-ONNXRuntime C++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
## In memory of Dr. Jian Sun
Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our remembrance and condolences to our captain Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center"><img src="assets/sunjian.png" width="200"></div>
没有孙剑博士的指导，YOLOX也不会问世并开源给社区使用。
孙剑博士的离去是CV领域的一大损失，我们在此特别添加了这个部分来表达对我们的“船长”孙老师的纪念和哀思。
希望世界上的每个AI从业者秉持着“持续创新拓展认知边界，非凡科技成就产品价值”的观念，一路向前。
