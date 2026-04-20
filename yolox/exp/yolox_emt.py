#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import json
import os

import torch.distributed as dist
import cv2
import numpy as np
from loguru import logger
from pycocotools.coco import COCO
import torch.nn as nn

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


class EMTDataset(CacheDataset):
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
            # Only count classes that survive the area filter
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
            if obj["area"] > 0 and x2 > x1 and y2 > y1 and  clipped_area >=  self.min_box_area:
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
        self.exp_name = "yolox_emt"

        self.num_classes = 3
        self.depth = 1
        self.width = 1

        self.data_dir = os.path.join(get_yolox_datadir(), "EMT")
        self.annotation_dir = "annotations/detections_new"
        self.train_ann = "train_3class.json"
        self.val_ann = "test_3class.json"
        self.test_ann = "test_3class.json"
        self.train_name = "frames"
        self.val_name = "frames"
        self.test_name = "frames"

        # Input sizes
        self.input_size = (1280, 1280)
        self.test_size = (1280, 1280)
        self.multiscale_range = 5

        # Training schedule
        self.max_epoch = 120
        self.print_interval = 50
        self.eval_interval = 2
        self.test_conf = 0.01
        self.nmsthre = 0.5
        self.no_aug_epochs = 20
        self.basic_lr_per_img = 0.001 / 64.0
        self.min_lr_ratio = 0.005
        self.warmup_epochs = 5

        # Augmentation
        self.enable_mixup = True
        self.mixup_prob = 0.5
        self.mosaic_prob = 0.8
        self.mosaic_scale = (0.5, 2.0)
        self.degrees = 5.0
        self.translate = 0.05
        self.shear = 0.5

        # Rare-class oversampling
        self.enable_rare_class_oversampling = True
        self.disable_oversampling_for_superclass = False
        self.auto_select_oversample_classes = False
        self.oversample_minority_ratio_threshold = 0.2
        self.oversample_target_classes = ("VulnerableRoadUser", "Two-Wheeler")
        self.max_oversample_factor = 4.0

        # Annotation filtering
        self.min_box_area = 75
        self.train_max_labels = 100
        self.mosaic_max_labels = 300

        # Class-weighted loss (order matches sorted category IDs)
        # index 0: VulnerableRoadUser (id 1)
        # index 1: Two-Wheeler        (id 2)
        # index 2: Vehicle            (id 3)
        self.cls_loss_weights = [2.0, 3.0, 1.0]

        # Logging
        self.print_class_stats_before_training = True
        self._printed_class_stats = False

    @staticmethod
    def _resolve_dataset_class_names(dataset_classes, requested_classes):
        if not dataset_classes or not requested_classes:
            return tuple()
        lookup = {name.strip().lower(): name for name in dataset_classes}
        resolved = []
        for class_name in requested_classes:
            matched = lookup.get(class_name.strip().lower())
            if matched is not None and matched not in resolved:
                resolved.append(matched)
        return tuple(resolved)

    @staticmethod
    def _is_main_process():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def _sync_num_classes_from_annotations(self):
        ann_path = os.path.join(self.data_dir, self.annotation_dir, self.train_ann)
        if not os.path.isfile(ann_path):
            return

        try:
            with open(ann_path, "r") as handle:
                payload = json.load(handle)
            categories = payload.get("categories", [])
            if categories:
                self.num_classes = len(categories)
                logger.info(
                    f"EMT exp detected {self.num_classes} classes from {ann_path}"
                )
        except Exception as exc:
            logger.warning(f"Failed to read {ann_path} for num_classes sync: {exc}")

    def get_dataset(self, cache=False, cache_type="ram"):
        return EMTDataset(
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
            min_box_area=self.min_box_area, 
        )
    
    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width,
                in_channels=in_channels,
                act=self.act
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                cls_loss_weights=self.cls_loss_weights,  
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        # Load dataset if not already created
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, (
                    "cache_img must be None if you didn't create self.dataset before launch"
                )
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        base_dataset = getattr(self.dataset, "_dataset", self.dataset)

        if (
            self.print_class_stats_before_training
            and not self._printed_class_stats
            and self._is_main_process()
            and hasattr(base_dataset, "get_class_statistics")
        ):
            annotation_count, image_count, filtered_count = base_dataset.get_class_statistics()
            logger.info("EMT class distribution before training:")
            for class_name in getattr(base_dataset, "_classes", tuple(annotation_count.keys())):
                total = annotation_count.get(class_name, 0)
                filtered = filtered_count.get(class_name, 0)
                kept = total - filtered
                logger.info(
                    f"  {class_name}: total={total}, "
                    f"filtered={filtered} ({100*filtered/max(total,1):.1f}%), "
                    f"kept={kept}, "
                    f"images={image_count.get(class_name, 0)}"
                )
            # Sanity check log — verify class weight alignment
            logger.info(f"Class order:    {base_dataset._classes}")
            logger.info(f"Class weights:  {self.cls_loss_weights}")
            self._printed_class_stats = True

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
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        # Adjust batch size for distributed training
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        dataset_classes = tuple(getattr(base_dataset, "_classes", ()))
        is_superclass_dataset = self._is_superclass_dataset(dataset_classes)
        use_oversampling = self.enable_rare_class_oversampling and not (
            self.disable_oversampling_for_superclass and is_superclass_dataset
        )

        if use_oversampling:
            target_classes = self._resolve_dataset_class_names(
                base_dataset._classes, self.oversample_target_classes
            )
            if self.auto_select_oversample_classes:
                _, image_frequency, _ = base_dataset.build_image_sampling_weights(
                    base_dataset._classes,
                    max_repeat_factor=self.max_oversample_factor,
                )
                max_freq = max(image_frequency.values()) if image_frequency else 1
                auto_selected = tuple(
                    name
                    for name in base_dataset._classes
                    if image_frequency.get(name, 0) > 0
                    and (image_frequency[name] / max_freq) <= self.oversample_minority_ratio_threshold
                )
                target_classes = tuple(dict.fromkeys(target_classes + auto_selected))

            if not target_classes:
                logger.warning(
                    "Oversampling enabled but none of the target classes exist in this dataset; "
                    "falling back to uniform sampling."
                )
                sampler = InfiniteSampler(
                    len(self.dataset), seed=self.seed if self.seed else 0
                )
            else:
                weights, image_frequency, repeat_factors = base_dataset.build_image_sampling_weights(
                    target_classes,
                    max_repeat_factor=self.max_oversample_factor,
                )
                expected_image_frequency = base_dataset.estimate_weighted_image_frequency(weights)
                logger.info(f"Oversample target classes: {target_classes}")
                logger.info(f"Using rare-class oversampling: {repeat_factors}")
                logger.info(
                    "Rare-class image frequencies: "
                    + str({name: image_frequency.get(name, 0) for name in target_classes})
                )
                logger.info("Expected sampled images per epoch after oversampling:")
                for class_name in base_dataset._classes:
                    raw_count = image_frequency.get(class_name, 0)
                    expected_count = expected_image_frequency.get(class_name, 0.0)
                    ratio = (expected_count / raw_count) if raw_count > 0 else 0.0
                    logger.info(
                        f"  {class_name}: raw_images={raw_count}, "
                        f"expected_images={expected_count:.1f}, x{ratio:.2f}"
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
            mosaic=not no_aug,
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
        return EMTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_name if not testdev else self.test_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            annotation_dir=self.annotation_dir,
            min_box_area=0,
        )
