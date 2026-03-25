import random
from math import ceil

import cv2
import numpy as np
import torch


class RoadDetrLoader2D(torch.utils.data.DataLoader):
    """
    Iterable loader for frame-level dataset (xyxy pixel boxes).
    If resize is set, scales boxes accordingly.
    """

    def __init__(self, dataset, batch_size=4, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_size = 720
        self.indices = list(range(len(dataset)))

        # ImageNet mean/std (numpy)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    def _resize_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(self.resize_size / max(h, w), 1.0)  # don’t go beyond 720
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def _prep_batch(self, items):
        images_paths, norm_images = [], []
        boxes_list, labels_list = [], []
        action_list, loc_list = [], []

        for image, tgt in items:
            images_paths.append(image[0])

            img_resized = self._resize_image(image[1])
            img_float = img_resized.astype(np.float32) / 255.0
            img_norm = (img_float - self.mean) / self.std
            img_norm_t = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # 3×H'×W'
            norm_images.append(img_norm_t)

            boxes_list.append(tgt["boxes"])
            labels_list.append(tgt["labels"])
            action_list.append(tgt["action_vec"])
            loc_list.append(tgt["loc_vec"])

        B = len(norm_images)
        heights = [im.shape[1] for im in norm_images]
        widths = [im.shape[2] for im in norm_images]
        H_max, W_max = max(heights), max(widths)

        images = norm_images[0].new_zeros((B, 3, H_max, W_max))
        mask = torch.ones((B, H_max, W_max), dtype=torch.bool)
        for i, im in enumerate(norm_images):
            _, h, w = im.shape
            images[i, :, :h, :w] = im
            mask[i, :h, :w] = False

        batch = {
            "images": images,  # [B,3,H_max,W_max]
            "mask": mask,  # [B,H_max,W_max], bool
            "images_paths": images_paths,
            "boxes": boxes_list,
            "labels": labels_list,
            "action_vec": action_list,
            "loc_vec": loc_list,
        }
        return batch

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            idxs = self.indices[i : i + self.batch_size]
            items = [self.dataset[j] for j in idxs]
            yield self._prep_batch(items)
