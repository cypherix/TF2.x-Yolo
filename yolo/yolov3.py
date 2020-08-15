#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:20:59 2020

@author: arjun
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    UpSampling2D,
   )

from .backbone import Darknet, DarknetConv
from .utils import ANCHORS, STRIDES


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(classes=80, masks=None, strides=None):
    def yolo_output(x):

        batch_size, output_size = tf.shape(x)[:2]
        x_output = tf.reshape(x, (-1, output_size, output_size,
                                  3, 5 + classes))

        x_dxdy = x_output[:, :, :, :, 0:2]
        x_dwdh = x_output[:, :, :, :, 2:4]
        x_conf = x_output[:, :, :, :, 4:5]
        x_prob = x_output[:, :, :, :, 5:]

        # Draw the grid

        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, axis=-1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size, dtype=tf.int32)
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]],
                            axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],
                          [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(x_dxdy) + xy_grid) * strides
        pred_wh = tf.exp(x_dwdh) * masks * strides

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(x_conf)
        pred_prob = tf.sigmoid(x_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    return yolo_output


def YoloV3(size=None, classes=80, training=False):
    x = inputs = Input([size, size, 3], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    l_output = DarknetConv(x, filters=3*(classes + 5),
                           size=1, activate=False, bn=False)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    m_output = DarknetConv(x, filters=3*(classes + 5),
                           size=1, activate=False, bn=False)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    s_output = DarknetConv(x, filters=3*(classes + 5),
                           size=1, activate=False, bn=False)

    output_tensors = []
    for i, output_tensor in enumerate([s_output, m_output, l_output]):
        pred_tensor = YoloOutput(classes, masks=ANCHORS[i],
                                 strides=STRIDES[i])(output_tensor)
        if training:
            output_tensors.append(output_tensor)
        output_tensors.append(pred_tensor)

    return Model(inputs, output_tensors)
