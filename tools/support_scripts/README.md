Support scripts for dataset preparation and inspection live here.

Available scripts:
- `emt_to_coco.py`: convert EMT KITTI-style tracking labels to COCO JSON for YOLOX.
- `road_to_coco.py`: convert ROAD Waymo annotations to COCO JSON for YOLOX.
- `visualize_annotations.py`: render KITTI or COCO labels on a sample image or a whole video folder.

Examples:
```bash
python tools/support_scripts/emt_to_coco.py
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
```
