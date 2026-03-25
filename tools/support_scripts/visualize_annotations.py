#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI TXT or COCO JSON annotations on a sample image or a whole video."
    )
    parser.add_argument(
        "--label-format",
        required=True,
        choices=("kitti", "coco"),
        help="Annotation format to read.",
    )
    parser.add_argument(
        "--labels-path",
        required=True,
        help="For KITTI: label txt file or directory. For COCO: annotation json file.",
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Root folder containing video frame directories.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sample", "video"),
        help="Visualize one frame or a full video folder.",
    )
    parser.add_argument("--video-name", help="Video folder name under images root.")
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
        default="visualizations",
        help="Where annotated outputs are written.",
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


def color_from_text(text):
    seed = sum(ord(ch) for ch in str(text))
    return (
        80 + (seed * 37) % 176,
        80 + (seed * 67) % 176,
        80 + (seed * 97) % 176,
    )


def parse_kitti_label_file(label_path):
    frame_to_annotations = {}
    with label_path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                class_name = parts[2]
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
            except ValueError as exc:
                raise ValueError(f"invalid line {line_num} in {label_path}") from exc

            frame_to_annotations.setdefault(frame_id, []).append(
                {
                    "class_name": class_name,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return frame_to_annotations


def load_kitti_annotations(labels_path, video_name):
    labels_path = Path(labels_path)
    label_file = labels_path if labels_path.is_file() else labels_path / f"{video_name}.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"label file not found: {label_file}")
    return parse_kitti_label_file(label_file)


def load_coco_annotations(labels_path, video_name=None):
    data = json.loads(Path(labels_path).read_text())
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    image_by_id = {}
    frame_to_annotations = {}

    for image in data.get("images", []):
        file_name = image["file_name"]
        image_video_name = Path(file_name).parent.name
        if video_name and image_video_name != video_name:
            continue
        frame_num = int(Path(file_name).stem)
        image_by_id[image["id"]] = frame_num

    for ann in data.get("annotations", []):
        frame_num = image_by_id.get(ann["image_id"])
        if frame_num is None:
            continue
        x1, y1, width, height = ann["bbox"]
        frame_to_annotations.setdefault(frame_num, []).append(
            {
                "class_name": categories.get(ann["category_id"], str(ann["category_id"])),
                "track_id": ann.get("track_id"),
                "bbox": [x1, y1, x1 + width, y1 + height],
            }
        )

    return frame_to_annotations


def build_frame_index(video_dir):
    frame_index = {}
    for image_path in sorted(video_dir.glob("*.jpg")):
        frame_index[int(image_path.stem)] = image_path
    return frame_index


def resolve_sample(args):
    if args.image_path:
        image_path = Path(args.image_path)
        video_name = args.video_name or image_path.parent.name
        frame_id = int(image_path.stem)
        return video_name, frame_id, image_path

    if not args.video_name or args.frame_id is None:
        raise ValueError(
            "sample mode requires either --image-path or both --video-name and --frame-id"
        )

    frame_index = build_frame_index(Path(args.images_root) / args.video_name)
    image_path = frame_index.get(args.frame_id)
    if image_path is None:
        raise FileNotFoundError(
            f"frame {args.frame_id} not found in {(Path(args.images_root) / args.video_name)}"
        )
    return args.video_name, args.frame_id, image_path


def draw_annotations(image, annotations):
    rendered = image.copy()
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        color = color_from_text(ann["class_name"])
        cv2.rectangle(rendered, p1, p2, color, 2)

        label = ann["class_name"]
        if ann.get("track_id") is not None:
            label = f"{label}#{ann['track_id']}"

        text_origin = (p1[0], max(18, p1[1] - 8))
        cv2.putText(
            rendered,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def write_sample(args, frame_to_annotations):
    video_name, frame_id, image_path = resolve_sample(args)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image {image_path}")

    annotations = frame_to_annotations.get(frame_id, [])
    rendered = draw_annotations(image, annotations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}_{frame_id:06d}.jpg"
    cv2.imwrite(str(output_path), rendered)
    print(f"sample written to {output_path}")
    print(f"annotations drawn: {len(annotations)}")


def write_video(args, frame_to_annotations):
    if not args.video_name:
        raise ValueError("video mode requires --video-name")

    video_dir = Path(args.images_root) / args.video_name
    frame_index = build_frame_index(video_dir)
    if not frame_index:
        raise ValueError(f"no frames found in {video_dir}")

    output_dir = Path(args.output_dir) / args.video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    video_path = Path(args.output_dir) / f"{args.video_name}.mp4"
    processed = 0

    for frame_id in sorted(frame_index):
        if args.limit is not None and processed >= args.limit:
            break

        image_path = frame_index[frame_id]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"warning: cannot read {image_path}")
            continue

        rendered = draw_annotations(image, frame_to_annotations.get(frame_id, []))
        frame_output = output_dir / image_path.name
        cv2.imwrite(str(frame_output), rendered)

        if args.save_video:
            if writer is None:
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
            writer.write(rendered)

        processed += 1

    if writer is not None:
        writer.release()
        print(f"video written to {video_path}")

    print(f"rendered {processed} frames to {output_dir}")


def main():
    args = parse_args()

    if args.label_format == "kitti":
        target_video = args.video_name
        if args.mode == "sample" and args.image_path and not target_video:
            target_video = Path(args.image_path).parent.name
        if not target_video:
            raise ValueError(
                "KITTI visualization requires --video-name or an image path inside a video folder"
            )
        frame_to_annotations = load_kitti_annotations(args.labels_path, target_video)
    else:
        target_video = args.video_name
        if args.mode == "sample" and args.image_path and not target_video:
            target_video = Path(args.image_path).parent.name
        frame_to_annotations = load_coco_annotations(args.labels_path, target_video)

    if args.mode == "sample":
        write_sample(args, frame_to_annotations)
    else:
        write_video(args, frame_to_annotations)


if __name__ == "__main__":
    main()
