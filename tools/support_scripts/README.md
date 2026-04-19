Support scripts for dataset preparation and inspection live here.

Available scripts:
- `emt_to_coco.py`: convert EMT KITTI-style tracking labels to COCO JSON for YOLOX.
- `emt_video_class_counts.py`: print per-video bbox totals for every class from EMT COCO JSON files.
- `road_to_coco.py`: convert ROAD Waymo annotations to COCO JSON for YOLOX.
- `swap_emt_videos.py`: move one video from train to test (or swap one each way), rebuild ids, and print train/test video lists.
- `visualize_annotations.py`: render KITTI or COCO labels on a sample image or a whole video folder.
- `visualize_emt.py`: EMT-focused wrapper around the visualizer with dataset defaults.

Examples:
```bash
python tools/support_scripts/emt_to_coco.py
python tools/support_scripts/emt_to_coco.py --dataset-root datasets/EMT --label-level superclass
python tools/support_scripts/emt_video_class_counts.py
python tools/support_scripts/emt_video_class_counts.py --splits train_class
python tools/support_scripts/swap_emt_videos.py \
  --ann-dir datasets/EMT/annotations/detections \
  --train-json train_class.json \
  --test-json test_class.json \
  --mode move \
  --train-video video_16
python tools/support_scripts/swap_emt_videos.py \
  --ann-dir datasets/EMT/annotations/detections \
  --train-json train_class.json \
  --test-json test_class.json \
  --mode move \
  --train-video video_16 \
  --apply
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo --match-emt
python tools/support_scripts/road_to_coco.py --road-dir datasets/road_waymo --no-match-emt
python tools/support_scripts/visualize_annotations.py \
  --label-format kitti \
  --labels-path datasets/emt/emt_annotations/labels_full \
  --images-root datasets/emt/frames \
  --mode sample \
  --video-name video_054604 \
  --frame-id 149
python tools/support_scripts/visualize_annotations.py \
  --label-format coco \
  --labels-path datasets/emt/emt_annotations/train.json \
  --images-root datasets/emt/frames \
  --mode video \
  --video-name video_054604 \
  --save-video
python tools/support_scripts/visualize_emt.py \
  --label-format coco \
  --split train \
  --mode sample \
  --video-name video_15\
  --frame-id 149
```
