#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from yolox.exp.yolox_emt_from_waymo import Exp as EMTwaymoExp


class Exp(EMTwaymoExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_emt_from_waymo"
