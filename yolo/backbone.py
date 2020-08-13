#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:50:37 2020

@author: arjun
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Input,
    Conv2D,
    LeakyReLU,
    ZeroPadding2D,
    BatchNormalization)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


class BatchNormalization(BatchNormalization):

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def DarknetConv(x, filters, size, downsample=False, activate=True, bn=True):

    if downsample:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding, use_bias=not bn,
               kernel_regularizer=l2(0.0005),
               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
               bias_initializer=tf.constant_initializer(0.))(x)

    if bn:
        x = BatchNormalization()(x)
    if activate:
        x = LeakyReLU(alpha=0.1)(x)

    return x


def DarknetResidual(x, filters):

    short_cut = x
    x = DarknetConv(x, filters=filters//2, size=1)
    x = DarknetConv(x, filters=filters, size=3)
    x = Add()([short_cut, x])

    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters=filters, size=3, downsample=True)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)

    return x


def Darknet(name=None):

    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, filters=32, size=3)
    x = DarknetBlock(x, 64, 2)
    x = DarknetBlock(x, 128, 2)
    x = x_36 = DarknetBlock(x, 256, 8)
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)

    return Model(inputs, (x_36, x_61, x), name=name)
