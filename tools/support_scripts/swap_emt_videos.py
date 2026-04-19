#!/usr/bin/env python3

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Move one video from EMT train->test (or swap one each way), "
            "then rebuild ids and write new annotation files."
        )
    )
    parser.add_argument(
        "--ann-dir",
        default="datasets/EMT/annotations/detections",
        help="Directory containing source json files.",
    )
    parser.add_argument("--train-json", default="train_class.json", help="Input train annotation json.")
    parser.add_argument("--test-json", default="test_class.json", help="Input test annotation json.")
    parser.add_argument(
        "--mode",
        choices=["move", "swap"],
        default="move",
        help="move: train->test only, swap: exchange one train and one test video.",
    )
    parser.add_argument("--train-video", required=True, help="Video currently in train split.")
    parser.add_argument(
        "--test-video",
        help="Video currently in test split. Required when --mode swap.",
    )
    parser.add_argument(
        "--out-train-json",
        default="newtrain.json",
        help="Output filename for rebuilt train annotations.",
    )
    parser.add_argument(
        "--out-test-json",
        default="newtest.json",
        help="Output filename for rebuilt test annotations.",
    )
    parser.add_argument(
        "--out-train-map",
        default="newtrain_mapping.json",
        help="Output filename for rebuilt train mapping json.",
    )
    parser.add_argument(
        "--out-test-map",
        default="newtest_mapping.json",
        help="Output filename for rebuilt test mapping json.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write output files. Without this flag, only print a dry-run summary.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_to_video_name(image_obj):
    file_name = image_obj.get("file_name", "")
    if isinstance(file_name, str) and "/" in file_name:
        return file_name.split("/", 1)[0]

    video_id = image_obj.get("video_id")
    if video_id is not None:
        return f"video_{video_id}"

    return "video_unknown"


def get_video_name(video_obj):
    if "folder_name" in video_obj:
        return video_obj["folder_name"]
    if "name" in video_obj:
        return video_obj["name"]
    if "file_name" in video_obj:
        return video_obj["file_name"]
    raise KeyError(f"Cannot infer video name from video object keys: {list(video_obj.keys())}")


def extract_video_names(split_data):
    videos = split_data.get("videos", [])
    if videos:
        return [get_video_name(v) for v in videos]

    seen = set()
    names = []
    for im in split_data.get("images", []):
        name = image_to_video_name(im)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def index_split(split_data):
    videos = split_data.get("videos", [])
    images = split_data.get("images", [])
    annotations = split_data.get("annotations", [])

    video_id_to_name = {}
    video_meta_by_name = {}
    video_order = []
    video_order_set = set()

    for v in videos:
        name = get_video_name(v)
        video_id_to_name[v["id"]] = name
        video_meta_by_name[name] = v
        if name not in video_order_set:
            video_order.append(name)
            video_order_set.add(name)

    images_by_video = defaultdict(list)
    for im in images:
        name = video_id_to_name.get(im.get("video_id"))
        if name is None:
            name = image_to_video_name(im)
            if name not in video_meta_by_name:
                video_meta_by_name[name] = {"folder_name": name}

        images_by_video[name].append(im)

        if name not in video_order_set:
            video_order.append(name)
            video_order_set.add(name)

    anns_by_image_id = defaultdict(list)
    for ann in annotations:
        anns_by_image_id[ann["image_id"]].append(ann)

    for name in images_by_video:
        images_by_video[name].sort(key=lambda x: (x.get("frame_id", 0), x["file_name"]))
        for im in images_by_video[name]:
            anns_by_image_id[im["id"]].sort(key=lambda a: a.get("id", 0))

    return {
        "video_order": video_order,
        "video_meta_by_name": video_meta_by_name,
        "images_by_video": images_by_video,
        "anns_by_image_id": anns_by_image_id,
        "categories": split_data.get("categories", []),
        "has_videos_key": "videos" in split_data,
    }


def build_video_payload(source_index, video_name):
    if video_name not in source_index["images_by_video"]:
        return None

    return {
        "video_meta": source_index["video_meta_by_name"].get(video_name, {"folder_name": video_name}),
        "images": source_index["images_by_video"][video_name],
        "anns_by_image_id": source_index["anns_by_image_id"],
    }


def collect_payloads(video_names_in_order, preferred_index, fallback_index, split_label):
    payloads = {}
    for v in video_names_in_order:
        payload = build_video_payload(preferred_index, v) or build_video_payload(fallback_index, v)
        if payload is None:
            raise ValueError(f"Cannot build payload for {split_label} video: {v}")
        payloads[v] = payload
    return payloads


def rebuild_split(video_names_in_order, payload_by_video, categories, include_videos):
    out = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    if include_videos:
        out["videos"] = []

    next_image_id = 1
    next_ann_id = 1

    for next_video_id, video_name in enumerate(video_names_in_order, start=1):
        payload = payload_by_video[video_name]

        if include_videos:
            video_meta = dict(payload["video_meta"])
            video_meta["id"] = next_video_id
            if "folder_name" not in video_meta:
                video_meta["folder_name"] = video_name
            out["videos"].append(video_meta)

        for im in payload["images"]:
            old_image_id = im["id"]
            anns_for_image = payload["anns_by_image_id"].get(old_image_id, [])

            new_im = dict(im)
            new_im["id"] = next_image_id
            if "video_id" in new_im or include_videos:
                new_im["video_id"] = next_video_id
            if "has_annotation" in new_im:
                new_im["has_annotation"] = len(anns_for_image) > 0
            out["images"].append(new_im)

            for ann in anns_for_image:
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                out["annotations"].append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    return out


def build_mapping(split_data):
    cat_id_to_name = {c["id"]: c["name"] for c in split_data.get("categories", [])}
    mapping = {}
    for ann in split_data.get("annotations", []):
        track_id_str = ann.get("track_id_str")
        if not track_id_str:
            continue
        class_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
        key = f"{track_id_str}|||{class_name}"
        if key not in mapping:
            mapping[key] = ann.get("track_id")
    return mapping


def summarize(name, split_data):
    video_names = extract_video_names(split_data)
    print(
        f"{name}: videos={len(video_names)} images={len(split_data.get('images', []))} "
        f"annotations={len(split_data.get('annotations', []))}"
    )
    print(f"{name} videos: {video_names}")


def move_video_train_to_test(train_data, test_data, train_video):
    train_idx = index_split(train_data)
    test_idx = index_split(test_data)

    if train_video not in train_idx["images_by_video"]:
        raise ValueError(f"{train_video} not found in train split.")
    if train_video in test_idx["images_by_video"]:
        raise ValueError(f"{train_video} is already in test split.")

    new_train_order = [v for v in train_idx["video_order"] if v != train_video]
    new_test_order = list(test_idx["video_order"]) + [train_video]

    payload_by_video_train = collect_payloads(new_train_order, train_idx, test_idx, "train")
    payload_by_video_test = collect_payloads(new_test_order, test_idx, train_idx, "test")

    new_train = rebuild_split(
        new_train_order,
        payload_by_video_train,
        train_data["categories"],
        include_videos=train_idx["has_videos_key"],
    )
    new_test = rebuild_split(
        new_test_order,
        payload_by_video_test,
        test_data["categories"],
        include_videos=test_idx["has_videos_key"],
    )
    return new_train, new_test


def swap_one_train_and_test_video(train_data, test_data, train_video, test_video):
    train_idx = index_split(train_data)
    test_idx = index_split(test_data)

    if train_video not in train_idx["images_by_video"]:
        raise ValueError(f"{train_video} not found in train split.")
    if test_video not in test_idx["images_by_video"]:
        raise ValueError(f"{test_video} not found in test split.")

    new_train_order = [v for v in train_idx["video_order"] if v != train_video] + [test_video]
    new_test_order = [v for v in test_idx["video_order"] if v != test_video] + [train_video]

    payload_by_video_train = collect_payloads(new_train_order, train_idx, test_idx, "train")
    payload_by_video_test = collect_payloads(new_test_order, test_idx, train_idx, "test")

    new_train = rebuild_split(
        new_train_order,
        payload_by_video_train,
        train_data["categories"],
        include_videos=train_idx["has_videos_key"],
    )
    new_test = rebuild_split(
        new_test_order,
        payload_by_video_test,
        test_data["categories"],
        include_videos=test_idx["has_videos_key"],
    )
    return new_train, new_test


def backup_if_exists(path):
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)

    train_path = ann_dir / args.train_json
    test_path = ann_dir / args.test_json

    out_train_path = ann_dir / args.out_train_json
    out_test_path = ann_dir / args.out_test_json
    out_train_map_path = ann_dir / args.out_train_map
    out_test_map_path = ann_dir / args.out_test_map

    train_data = load_json(train_path)
    test_data = load_json(test_path)

    if train_data.get("categories") != test_data.get("categories"):
        raise ValueError("Train/test categories differ. Refusing to proceed.")

    if args.mode == "swap" and not args.test_video:
        raise ValueError("--test-video is required when --mode swap")

    if args.mode == "move":
        new_train, new_test = move_video_train_to_test(train_data, test_data, args.train_video)
    else:
        new_train, new_test = swap_one_train_and_test_video(
            train_data, test_data, args.train_video, args.test_video
        )

    new_train_map = build_mapping(new_train)
    new_test_map = build_mapping(new_test)

    print("Plan:")
    if args.mode == "move":
        print(f"  train -> test: {args.train_video}")
    else:
        print(f"  train -> test: {args.train_video}")
        print(f"  test  -> train: {args.test_video}")
    print("")
    summarize("old_train", train_data)
    summarize("old_test", test_data)
    print("")
    summarize("new_train", new_train)
    summarize("new_test", new_test)
    print("")
    print(f"new_train_mapping keys: {len(new_train_map)}")
    print(f"new_test_mapping keys: {len(new_test_map)}")
    print("")
    print(f"output_train_json: {out_train_path}")
    print(f"output_test_json: {out_test_path}")
    print(f"output_train_map: {out_train_map_path}")
    print(f"output_test_map: {out_test_map_path}")

    if not args.apply:
        print("\nDry run only. Use --apply to write files.")
        return

    for p in [out_train_path, out_test_path, out_train_map_path, out_test_map_path]:
        backup = backup_if_exists(p)
        if backup is not None:
            print(f"backup: {backup}")

    write_json(out_train_path, new_train)
    write_json(out_test_path, new_test)
    write_json(out_train_map_path, new_train_map)
    write_json(out_test_map_path, new_test_map)

    print("\nApplied.")
    print(f"wrote: {out_train_path}")
    print(f"wrote: {out_test_path}")
    print(f"wrote: {out_train_map_path}")
    print(f"wrote: {out_test_map_path}")


if __name__ == "__main__":
    main()
