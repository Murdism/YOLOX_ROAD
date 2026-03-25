import json
import os
import re
from pathlib import Path

import cv2
import torch


class RoadDetrDataset2D(torch.utils.data.Dataset):
    """
    Frame-level ROAD dataset (bbox + agent only).
    Returns:
      image:  raw cv2 image [H,W,3] (BGR)
      target: dict {
          'boxes':      Tensor [N,4]           # xyxy normalized in [0,1]
          'labels':     Tensor [N]             # agent class ids
          'action_vec': Tensor [N,K_action]    # multi-hot action labels
          'loc_vec':    Tensor [N,K_loc]       # multi-hot location labels
      }
    """

    def __init__(self, dataset_dir, prefix):
        if prefix in ["train", "val"]:
            self.dataset_dir = os.path.join(dataset_dir, "train")
        else:
            self.dataset_dir = os.path.join(dataset_dir, "test")
        self.prefix = prefix

        # load JSON
        files = sorted(Path(self.dataset_dir).glob("*.json"))
        if len(files) == 0:
            print(
                f"json not found, verify dataset path {self.dataset_dir} contains json"
            )

        self.ann_json = files[0]
        with open(self.ann_json, "r") as f:
            ann_data = json.load(f)
            self.agent_labels = ann_data["all_agent_labels"]
            self.action_labels = ann_data["all_action_labels"]
            self.loc_labels = ann_data["all_loc_labels"]
            self.db = ann_data["db"]

        self.frames_dir = os.path.join(self.dataset_dir, "rgb-images")
        self.samples = self._gather_samples()

    def _num_from_name(self, name: str):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else None

    def _build_id2path(self, video_dir):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        id2path = {}
        for fn in os.listdir(video_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            fid = self._num_from_name(fn)
            if fid is not None:
                id2path[fid] = os.path.join(video_dir, fn)
        return id2path

    def _video_split_matching(self, split_ids):
        if self.prefix not in split_ids:
            if len(split_ids) == 4:
                split = split_ids[3].split("_")[0]
                if self.prefix != split:
                    return False
            else:
                return False
        return True

    def _fetch_video_frames_dir(self, vname):
        p1 = os.path.join(self.frames_dir, f"videos_{vname}")
        p2 = os.path.join(self.frames_dir, vname)
        return p1 if os.path.isdir(p1) else p2

    def _gather_samples(self):
        samples = []
        for vname, entry in self.db.items():
            if not self._video_split_matching(entry["split_ids"]):
                continue
            video_frames_dir = self._fetch_video_frames_dir(vname)
            if not os.path.isdir(video_frames_dir):
                continue
            id2path = self._build_id2path(video_frames_dir)
            for fid_str, frame_data in entry["frames"].items():
                fid = int(fid_str)
                if fid not in id2path:
                    continue
                annos = frame_data.get("annos", {})
                if not annos:
                    continue
                samples.append({"image_path": id2path[fid], "annotations": annos})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")
        image = (image_path, img)

        boxes, labels = [], []
        action_vecs, loc_vecs = [], []

        K_action = len(self.action_labels)
        K_loc = len(self.loc_labels)

        for _, obj in sample["annotations"].items():
            x1, y1, x2, y2 = obj["box"]
            agent_id = obj["agent_ids"][0]

            boxes.append([x1, y1, x2, y2])
            labels.append(agent_id)

            # multi-hot action vector
            a_vec = torch.zeros(K_action, dtype=torch.float16)
            for aid in obj.get("action_ids", []):
                if 0 <= aid < K_action:
                    a_vec[aid] = 1.0
            action_vecs.append(a_vec)

            # multi-hot location vector
            l_vec = torch.zeros(K_loc, dtype=torch.float16)
            for lid in obj.get("loc_ids", []):
                if 0 <= lid < K_loc:
                    l_vec[lid] = 1.0
            loc_vecs.append(l_vec)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float16),
            "labels": torch.tensor(labels, dtype=torch.long),
            "action_vec": (
                torch.stack(action_vecs, dim=0)
                if action_vecs
                else torch.zeros((0, K_action), dtype=torch.float16)
            ),
            "loc_vec": (
                torch.stack(loc_vecs, dim=0)
                if loc_vecs
                else torch.zeros((0, K_loc), dtype=torch.float16)
            ),
            "image_path": sample["image_path"],  # include path for on-demand loading
        }

        return image, target
