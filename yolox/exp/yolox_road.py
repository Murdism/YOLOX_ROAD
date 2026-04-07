#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import os

import torch.distributed as dist
import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.data import (
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    TrainTransform,
    ValTransform,
    WeightedInfiniteSampler,
    YoloBatchSampler,
    get_yolox_datadir,
    worker_init_reset_seed,
)
from yolox.data.datasets.coco import remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img
from yolox.exp.yolox_base import Exp as YOLOXBaseExp
from yolox.utils import wait_for_the_master


class RoadDataset(CacheDataset):
    def __init__(
        self,
        data_dir=None,
        json_file="train.json",
        name="train_frames",
        img_size=(960, 960),
        preproc=None,
        cache=False,
        cache_type="ram",
        annotation_dir="road_waymo_annotations",
    ):
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "road_waymo")

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotation_dir = annotation_dir

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
            image_class_sets.append(
                {self.cat_id_to_name[ann["category_id"]] for ann in anns}
            )
        return image_class_sets

    def build_image_sampling_weights(self, target_class_names, max_repeat_factor=8.0):
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
            repeat_factors[class_name] = float(
                min(max_repeat_factor, max(1.0, np.sqrt(max_image_frequency / freq)))
            )

        weights = np.ones(self.num_imgs, dtype=np.float32)
        for idx, class_names in enumerate(self.image_class_sets):
            matched = [repeat_factors[name] for name in target_class_names if name in class_names]
            if matched:
                weights[idx] = max(matched)

        return weights, image_frequency, repeat_factors

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
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
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


class Exp(YOLOXBaseExp):
    def __init__(self):
        super().__init__()

        self.output_dir = "./checkpoints"
        self.exp_name = "yolox_road_waymo"

        self.num_classes = 9  # adjust for your dataset
        self.depth = 1.33
        self.width = 1.25

        self.data_dir = os.path.join(get_yolox_datadir(), "road_waymo")
        self.annotation_dir = "road_waymo_annotations"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_name = "train_frames"
        self.val_name = "train_frames"
        self.test_name = "train_frames"

        self.input_size = (960, 1280)
        self.test_size = (960, 1280)

        self.multiscale_range = 0
        # remove self.random_size

        self.basic_lr_per_img = 0.01 / 64.0
        self.warmup_epochs = 5
        self.no_aug_epochs = 15

        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.mosaic_prob = 0.3
        self.mosaic_scale = (0.8, 1.2)
        self.degrees = 2.0
        self.translate = 0.03
        self.shear = 0.5

        self.enable_rare_class_oversampling = True
        self.oversample_target_classes = (
            "Emergency_vehicle",
            "Small_motorised_vehicle",
            "Cyclist",
        )
        self.max_oversample_factor = 6.0

        self.test_conf = 0.01
        self.nmsthre = 0.65
    def get_dataset(self, cache=False, cache_type="ram"):
        return RoadDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.train_max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
            annotation_dir=self.annotation_dir,
        )
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        # Load dataset if not already created
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, (
                    "cache_img must be None if you didn't create self.dataset before launch"
                )
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        base_dataset = getattr(self.dataset, "_dataset", self.dataset)

        # Wrap with MosaicDetection for augmentations
        self.dataset = MosaicDetection(
            dataset=base_dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=self.mosaic_max_labels,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,   # fixed scale applied
            mixup_scale=self.mixup_scale,     # fixed safe range
            shear=self.shear,
            enable_mixup=self.enable_mixup,   # MixUp on/off
            mosaic_prob=self.mosaic_prob,     # Mosaic probability
            mixup_prob=self.mixup_prob,       # MixUp probability
        )

        # Adjust batch size for distributed training
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        if self.enable_rare_class_oversampling:
            weights, image_frequency, repeat_factors = base_dataset.build_image_sampling_weights(
                self.oversample_target_classes,
                max_repeat_factor=self.max_oversample_factor,
            )
            print(f"Using rare-class oversampling: {repeat_factors}")
            print(
                "Rare-class image frequencies: "
                + str({name: image_frequency.get(name, 0) for name in self.oversample_target_classes})
            )
            sampler = WeightedInfiniteSampler(
                weights, seed=self.seed if self.seed else 0
            )
        else:
            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,  # Mosaic enabled during batch sampling
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }

        return DataLoader(self.dataset, **dataloader_kwargs)
    def get_eval_dataset(self, **kwargs):
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        return RoadDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_name if not testdev else self.test_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            annotation_dir=self.annotation_dir,
        )
