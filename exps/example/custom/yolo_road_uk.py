#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from yolox.exp.yolox_road_uk import Exp as RoadUkExp


class Exp(RoadUkExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_road_uk"