#!/usr/bin/env python3

import argparse
from pathlib import Path
from types import SimpleNamespace

from visualize_annotations import (
    load_coco_annotations,
    load_kitti_annotations,
    write_sample,
    write_video,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize EMT annotations on one frame or a whole video."
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/EMT",
        help="EMT dataset root (for example datasets/EMT).",
    )
    parser.add_argument(
        "--annotation-dir",
        default="annotations/detections/",
        help=(
            "Optional annotation directory under dataset root. "
            "If omitted, the script auto-detects common EMT layouts."
        ),
    )
    parser.add_argument(
        "--label-format",
        choices=("kitti", "coco"),
        default="coco",
        help="Which EMT annotation format to read.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Optional explicit labels path. Overrides the default path for the chosen format.",
    )
    parser.add_argument(
        "--images-root",
        default=None,
        help="Optional explicit images root. Overrides the default split image directory.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sample", "video"),
        help="Visualize a single frame or an entire video folder.",
    )
    parser.add_argument("--video-name", help="Video folder name under the split image root.")
    parser.add_argument(
        "--frame-id",
        type=int,
        help="Frame id for sample mode, for example 149.",
    )
    parser.add_argument(
        "--image-path",
        help="Optional direct image path for sample mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/emt",
        help="Where rendered outputs are written.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="When mode=video, also write an mp4 file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS to use for saved mp4 output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of frames to render in video mode.",
    )
    return parser.parse_args()


def resolve_default_paths(args):
    dataset_root = Path(args.dataset_root)
    annotation_root = dataset_root / args.annotation_dir if args.annotation_dir else None

    if args.images_root is not None:
        images_root = Path(args.images_root)
    else:
        # New EMT layout uses frames/ for both train and test videos.
        # Keep backward compatibility for datasets with test_frames/.
        if (dataset_root / "frames").exists():
            images_root = dataset_root / "frames"
        elif args.split == "test" and (dataset_root / "test_frames").exists():
            images_root = dataset_root / "test_frames"
        else:
            images_root = dataset_root / "frames"

    if args.labels_path is not None:
        labels_path = Path(args.labels_path)
    elif args.label_format == "kitti":
        candidates = []
        if annotation_root is not None:
            candidates.extend([annotation_root / "labels_full", annotation_root])
        candidates.extend(
            [
                dataset_root / "annotations" / "tracking" / "kitti",
                dataset_root / "annotations" / "tracking" / "gmot",
                dataset_root / "emt_annotations" / "labels_full",
            ]
        )
        labels_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        json_name = "train.json" if args.split == "train" else "test.json"
        candidates = []
        if annotation_root is not None:
            candidates.append(annotation_root / json_name)
        candidates.extend(
            [
                dataset_root / "annotations" / "detections" / json_name,
                dataset_root / "emt_annotations" / json_name,
            ]
        )
        labels_path = next((p for p in candidates if p.exists()), candidates[0])

    return labels_path, images_root


def main():
    args = parse_args()
    labels_path, images_root = resolve_default_paths(args)

    render_args = SimpleNamespace(
        label_format=args.label_format,
        labels_path=str(labels_path),
        images_root=str(images_root),
        mode=args.mode,
        video_name=args.video_name,
        frame_id=args.frame_id,
        image_path=args.image_path,
        output_dir=args.output_dir,
        save_video=args.save_video,
        fps=args.fps,
        limit=args.limit,
    )

    if args.label_format == "kitti":
        target_video = args.video_name
        if args.mode == "sample" and args.image_path and not target_video:
            target_video = Path(args.image_path).parent.name
        if not target_video:
            raise ValueError(
                "KITTI visualization requires --video-name or an image path inside a video folder"
            )
        frame_to_annotations = load_kitti_annotations(labels_path, target_video)
    else:
        target_video = args.video_name
        if args.mode == "sample" and args.image_path and not target_video:
            target_video = Path(args.image_path).parent.name
        frame_to_annotations = load_coco_annotations(labels_path, target_video)

    if args.mode == "sample":
        write_sample(render_args, frame_to_annotations)
    else:
        write_video(render_args, frame_to_annotations)


if __name__ == "__main__":
    main()
