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


def iou(bboxes1, bboxes2):

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1 = tf.concat([bboxes1[..., 0:2] - bboxes1[..., 2:4] * 0.5,
                         bboxes1[..., 0:2] + bboxes1[..., 2:4] * 0.5], axis=-1)
    bboxes2 = tf.concat([bboxes2[..., 0:2] - bboxes2[..., 2:4] * 0.5,
                         bboxes2[..., 0:2] + bboxes2[..., 2:4] * 0.5], axis=-1)

    inter_box = tf.concat([tf.maximum(bboxes1[..., 0:2], bboxes2[..., 0:2]),
                           tf.minimum(bboxes1[..., 2:4], bboxes2[..., 2:4])],
                          axis=-1)
    inter_area = tf.maximum(
        (inter_box[..., 2] - inter_box[..., 0]) * (inter_box[..., 3] - inter_box[..., 1]), 0.0)

    union_area = bboxes1_area + bboxes2_area - inter_area
    return inter_area / union_area


def giou(bboxes1, bboxes2):

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1 = tf.concat([bboxes1[..., 0:2] - bboxes1[..., 2:4] * 0.5,
                         bboxes1[..., 0:2] + bboxes1[..., 2:4] * 0.5], axis=-1)
    bboxes2 = tf.concat([bboxes2[..., 0:2] - bboxes2[..., 2:4] * 0.5,
                         bboxes2[..., 0:2] + bboxes2[..., 2:4] * 0.5], axis=-1)

    inter_box = tf.concat([tf.maximum(bboxes1[..., 0:2], bboxes2[..., 0:2]),
                           tf.minimum(bboxes1[..., 2:4], bboxes2[..., 2:4])],
                          axis=-1)
    inter_area = tf.maximum(
        (inter_box[..., 2] - inter_box[..., 0]) * (inter_box[..., 3] - inter_box[..., 1]), 0.0)

    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = inter_area / union_area

    g_box = tf.concat([tf.minimum(bboxes1[..., 0:2], bboxes2[..., 0:2]),
                       tf.maximum(bboxes1[..., 2:4], bboxes2[..., 2:4])],
                      axis=-1)
    g_area = tf.maximum(
        (g_box[..., 2] - g_box[..., 0]) * (g_box[..., 3] - g_box[..., 1]), 0.0)
    return iou - (g_area - union_area) / g_area
