#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import os

import torch.distributed as dist
import torch.nn as nn
from loguru import logger

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
from yolox.data.datasets.emt_dataset import EMTDataset
from yolox.exp.yolox_base import Exp as YOLOXBaseExp
from yolox.utils import wait_for_the_master


class Exp(YOLOXBaseExp):
    """Fine-tuning config: initialise from a ROAD-Waymo checkpoint, train on EMT.

    Uses gentler augmentation, a lower learning rate, and shorter schedule
    compared to yolox_emt.py (the model is already mostly trained).
    """

    def __init__(self):
        super().__init__()

        self.output_dir = "./checkpoints"
        self.exp_name = "yolox_emt_from_waymo"

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

        # Input
        self.input_size = (1280, 1280)
        self.test_size = (1280, 1280)
        self.multiscale_range = 5

        # Shorter schedule — model already mostly trained
        self.max_epoch = 30
        self.print_interval = 50
        self.eval_interval = 1
        self.test_conf = 0.01
        self.nmsthre = 0.5
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.0001 / 64.0   # 10× lower than yolox_emt
        self.min_lr_ratio = 0.01
        self.warmup_epochs = 1

        # Gentler augmentation
        self.enable_mixup = True
        self.mixup_prob = 0.3
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.7, 1.5)
        self.degrees = 3.0
        self.translate = 0.05
        self.shear = 0.5

        # Milder imbalance handling
        self.enable_rare_class_oversampling = True
        self.auto_select_oversample_classes = False
        self.oversample_minority_ratio_threshold = 0.2
        self.oversample_target_classes = ("VulnerableRoadUser", "Two-Wheeler")
        self.max_oversample_factor = 3.0

        # Annotation filtering
        self.min_box_area = 75
        self.train_max_labels = 100
        self.mosaic_max_labels = 300

        # Gentler class weights (model already learned from Waymo)
        self.cls_loss_weights = [1.5, 2.0, 1.0]

        self.print_class_stats_before_training = True
        self._printed_class_stats = False

        self._sync_num_classes_from_annotations()

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
                logger.info(f"EMT exp detected {self.num_classes} classes from {ann_path}")
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
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
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
            logger.info(f"Class order:    {base_dataset._classes}")
            logger.info(f"Class weights:  {self.cls_loss_weights}")
            self._printed_class_stats = True

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

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        if self.enable_rare_class_oversampling:
            target_classes = self._resolve_dataset_class_names(
                base_dataset._classes, self.oversample_target_classes
            )
            if self.auto_select_oversample_classes:
                _, image_frequency, _ = base_dataset.build_image_sampling_weights(
                    base_dataset._classes, max_repeat_factor=self.max_oversample_factor,
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
                sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
            else:
                weights, image_frequency, repeat_factors = base_dataset.build_image_sampling_weights(
                    target_classes, max_repeat_factor=self.max_oversample_factor,
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
