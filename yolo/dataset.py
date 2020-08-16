#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:51:37 2020

@author: arjun
"""

import cv2
import numpy as np

from .utils import STRIDES, ANCHORS
from .utils import image_preprocess, bbox_iou


class Dataset(object):

    def __init__(self, df, num_classes, batch_size,
                 input_size=416, loadimg=False, augment=False):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.loadimg = loadimg
        self.data_aug = augment

        self.strides = STRIDES
        self.anchors = ANCHORS
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 100
        self.output_size = self.input_size // STRIDES

        self.annotations = self.load_annotations(df, self.loadimg)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, df, loadimg=False):

        annotations = []
        for img_path in df.img_path.unique():
            tmp = []
            for _, line in df[df.img_path == img_path].iterrows():
                rsa = line['region_shape_attributes']
                xmin = rsa['x']
                ymin = rsa['y']
                xmax = rsa['x'] + rsa['width']
                ymax = rsa['y'] + rsa['height']
                class_ = line['name']
                tmp.append([xmin, ymin, xmax, ymax, class_])

            annotations.append([img_path, tmp])
        return annotations

    def preprocess(self, annotaion):
        image, bboxes = annotaion
        bboxes = np.array(list(map(np.array, bboxes)))
        image, bboxes = image_preprocess(np.copy(cv2.imread(image)),
                                         target_size=self.input_size,
                                         gt_boxes=np.copy(bboxes))

        label = [np.zeros((self.output_size[i],
                           self.output_size[i],
                           self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes,
                                           1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]],
                axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / \
                self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :],
                                     anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(
                        bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

        label_sbbox, label_mbbox, label_lbbox = label
        return image, label_sbbox, label_mbbox, label_lbbox

    def __repr__(self):
        return (f'Dataset Generator with {self.num_samples} datapoints '
                'in {self.num_batchs} batches')

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        batch_images = np.zeros((self.batch_size, self.input_size,
                                 self.input_size, 3), dtype=np.float32)
        batch_slabels = np.zeros(
            (self.batch_size, self.output_size[0],
             self.output_size[0], 3, 5 + self.num_classes),
            dtype=np.float32)
        batch_mlabels = np.zeros(
            (self.batch_size, self.output_size[1],
             self.output_size[1], 3, 5 + self.num_classes),
            dtype=np.float32)
        batch_llabels = np.zeros(
            (self.batch_size, self.output_size[2],
             self.output_size[2], 3, 5 + self.num_classes),
            dtype=np.float32)

        if self.batch_count < self.num_batchs:
            for i in range(self.batch_size):
                idx = self.batch_count * self.batch_size + i
                if idx > self.num_samples:
                    idx = idx - self.num_samples
                annotation = self.annotations[idx]

                image, slabel, mlabel, llabel = self.preprocess(annotation)
                batch_images[i, ...] = image
                batch_slabels[i, ...] = slabel
                batch_mlabels[i, ...] = mlabel
                batch_llabels[i, ...] = llabel

            self.batch_count += 1
            return batch_images, [batch_slabels, batch_mlabels, batch_llabels]
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration
