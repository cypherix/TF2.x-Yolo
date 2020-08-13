#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:20:59 2020

@author: arjun
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensoflow.keras.layers import (
    Input,
    Concatenate,
    UpSampling2D,
   )

from .backbone import Darknet, DarknetConv


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


def YoloV3(size=None, classes=80):
    x = Input([size, size, 3], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    l_output = DarknetConv(x, filters=3*(classes + 5), size=1,
                           activate=False, bn=False)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    m_output = DarknetConv(x, filters=3*(classes + 5), size=1,
                           activate=False, bn=False)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    s_output = DarknetConv(x, filters=3*(classes + 5), size=1,
                           activate=False, bn=False)

    return [l_output, m_output, s_output]

