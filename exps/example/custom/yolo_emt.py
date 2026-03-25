#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from yolox.exp.yolox_emt import Exp as EMTExp


class Exp(EMTExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_emt"
