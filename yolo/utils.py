# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


# Some important constants
YOLO_STRIDES = [8, 16, 32]
YOLO_ANCHOR_PER_SCALE = 3
YOLO_INPUT_SIZE = 416
YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T
