#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.data import get_yolox_datadir
from yolox.data.datasets.coco import remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img


class EMTDataset(CacheDataset):
    """COCO-format dataset loader for the EMT road-scene dataset.

    Handles area-filtered annotation loading, per-image class tracking for
    rare-class oversampling, and optional RAM/disk image caching.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train.json",
        name="frames",
        img_size=(1280, 1280),
        preproc=None,
        cache=False,
        cache_type="ram",
        annotation_dir="emt_annotations",
        min_box_area=75,
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "emt")

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotation_dir = annotation_dir
        self.min_box_area = min_box_area

        ann_path = os.path.join(self.data_dir, self.annotation_dir, self.json_file)
        self.coco = COCO(ann_path)
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.cats}
        self.annotations = self._load_coco_annotations()
        self.image_class_sets = self._build_image_class_sets()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _build_image_class_sets(self):
        image_class_sets = []
        for img_id in self.ids:
            anns = self.coco.imgToAnns.get(img_id, [])
            valid_classes = set()
            for ann in anns:
                if ann.get("area", 0) >= self.min_box_area:
                    class_name = self.cat_id_to_name.get(ann["category_id"])
                    if class_name:
                        valid_classes.add(class_name)
            image_class_sets.append(valid_classes)
        return image_class_sets

    def build_image_sampling_weights(self, target_class_names, max_repeat_factor=10.0):
        if not target_class_names:
            return np.ones(self.num_imgs, dtype=np.float32), {}, {}

        image_frequency = {class_name: 0 for class_name in self._classes}
        for class_names in self.image_class_sets:
            for class_name in class_names:
                image_frequency[class_name] += 1

        max_image_frequency = max(image_frequency.values()) if image_frequency else 1
        repeat_factors = {}
        for class_name in target_class_names:
            freq = image_frequency.get(class_name, 0)
            if freq <= 0:
                repeat_factors[class_name] = 1.0
                continue
            raw_factor = max_image_frequency / freq
            repeat_factors[class_name] = float(min(max_repeat_factor, max(1.0, raw_factor)))

        weights = np.ones(self.num_imgs, dtype=np.float32)
        for idx, class_names in enumerate(self.image_class_sets):
            matched = [repeat_factors[name] for name in target_class_names if name in class_names]
            if matched:
                weights[idx] = max(matched)

        return weights, image_frequency, repeat_factors

    def get_class_statistics(self):
        annotation_count = {class_name: 0 for class_name in self._classes}
        image_count = {class_name: 0 for class_name in self._classes}
        filtered_count = {class_name: 0 for class_name in self._classes}

        for img_id in self.ids:
            anns = self.coco.imgToAnns.get(img_id, [])
            seen_classes = set()
            for ann in anns:
                class_name = self.cat_id_to_name.get(ann["category_id"])
                if class_name is None:
                    continue
                annotation_count[class_name] += 1
                if ann.get("area", 0) < self.min_box_area:
                    filtered_count[class_name] += 1
                else:
                    seen_classes.add(class_name)
            for class_name in seen_classes:
                image_count[class_name] += 1

        return annotation_count, image_count, filtered_count

    def estimate_weighted_image_frequency(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != self.num_imgs:
            raise ValueError("weights must be a 1D array with length == number of images")
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights must contain positive values")

        probs = weights / total
        expected_draws = {class_name: 0.0 for class_name in self._classes}
        for idx, class_names in enumerate(self.image_class_sets):
            p = float(probs[idx])
            if p <= 0.0:
                continue
            for class_name in class_names:
                expected_draws[class_name] += p

        draws_per_epoch = float(self.num_imgs)
        for class_name in expected_draws:
            expected_draws[class_name] *= draws_per_epoch
        return expected_draws

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = max(0, obj["bbox"][0])
            y1 = max(0, obj["bbox"][1])
            x2 = min(width, x1 + max(0, obj["bbox"][2]))
            y2 = min(height, y1 + max(0, obj["bbox"][3]))
            clipped_area = (x2 - x1) * (y2 - y1)
            if obj["area"] > 0 and x2 > x1 and y2 > y1 and clipped_area >= self.min_box_area:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        res = np.zeros((len(objs), 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann["file_name"]
        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])
