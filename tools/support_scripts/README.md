# Support Scripts

Dataset preparation and inspection tools for the YOLOX-ROAD fork.

---

## Conversion

### `emt_to_coco.py`
Converts EMT KITTI-style tracking labels to COCO JSON.

```bash
# Default paths (reads from datasets/EMT/emt_annotations/labels_full)
python tools/support_scripts/emt_to_coco.py

# Override root and use superclass labels
python tools/support_scripts/emt_to_coco.py \
  --dataset-root datasets/EMT \
  --label-level superclass
```

Outputs `train_3class.json` and `test_3class.json` (or `*_superclass.json`) to the annotation directory.

### `road_to_coco.py`
Converts ROAD Waymo source annotations to COCO JSON.

```bash
# Default — maps to EMT 3-class taxonomy
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo

# Keep original ROAD class names
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo --no-match-emt
```

### `road_waymo_tool.py`
Class stats, interactive visualization, and class remapping for ROAD Waymo. See the main README for full usage.

```bash
python tools/support_scripts/road_waymo_tool.py stats --split train
python tools/support_scripts/road_waymo_tool.py visualize --split train --video train_00444
python tools/support_scripts/road_waymo_tool.py merge --split train --preset emt \
    --out datasets/road_waymo/road_waymo_annotations/train_3class.json
```

### `emt_to_superclass.py`
Maps fine-grained EMT class labels to the 4-class superclass scheme (Motorbike, Pedestrian, Vehicle, Cyclist).

```bash
python tools/support_scripts/emt_to_superclass.py
```

### `road_uk_coco.py`
Converts ROAD UK annotations to COCO JSON.

```bash
python tools/support_scripts/road_uk_coco.py --road-dir datasets/road_uk
```

---

## Inspection

### `visualize_annotations.py`
Renders KITTI TXT or COCO JSON bounding boxes on frames. Supports single-frame preview and full-video rendering.

```bash
# Single frame — KITTI labels
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path datasets/emt/emt_annotations/labels_full \
  --images-root datasets/emt/frames \
  --mode sample \
  --video-name video_054604 \
  --frame-id 149

# Single frame — COCO JSON
python tools/support_scripts/visualize_annotations.py \
  --label-format coco \
  --labels-path datasets/emt/emt_annotations/train.json \
  --images-root datasets/emt/frames \
  --mode sample \
  --video-name video_054604 \
  --frame-id 149

# Full video — render and save
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path datasets/emt/emt_annotations/labels_full \
  --images-root datasets/emt/frames \
  --mode video \
  --video-name video_054604 \
  --save-video
```

### `visualize_emt.py`
EMT-specific wrapper around `visualize_annotations.py` with dataset defaults pre-filled.

```bash
python tools/support_scripts/visualize_emt.py \
  --label-format coco \
  --split train \
  --mode sample \
  --video-name video_15 \
  --frame-id 149
```

### `emt_video_class_counts.py`
Prints per-video bounding-box counts for every class from the EMT COCO JSON files.

```bash
python tools/support_scripts/emt_video_class_counts.py
python tools/support_scripts/emt_video_class_counts.py --splits train_class
```

### `preview_road_json.py`
Quick inspection of a ROAD Waymo JSON file — prints image and annotation counts by split.

```bash
python tools/support_scripts/preview_road_json.py
```

---

## Dataset Management

### `swap_emt_videos.py`
Moves one video from train to test (or swaps one each way), rebuilds COCO image IDs, and prints the resulting split lists.

```bash
# Dry run — show what would change
python tools/support_scripts/swap_emt_videos.py \
  --ann-dir datasets/EMT/annotations/detections \
  --train-json train_class.json \
  --test-json test_class.json \
  --mode move \
  --train-video video_16

# Apply the change
python tools/support_scripts/swap_emt_videos.py \
  --ann-dir datasets/EMT/annotations/detections \
  --train-json train_class.json \
  --test-json test_class.json \
  --mode move \
  --train-video video_16 \
  --apply
```
