#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python
# !-*-coding:utf-8-*-

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from .utils import transform


class PascalVOCDataset(Dataset):
    def __init__(self, data_dir: str, split: str, keep_difficult=False):
        """

        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_dir
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_dir, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_dir, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, index):
        image = Image.open(self.images[index], mode='r')
        image = image.convert('RGB')

        objects = self.objects[index]
        boxes = torch.FloatTensor(objects['boxes'])     #(n_objects, 4)
        labels = torch.FloatTensor(objects['labels'])   #(n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])    #(n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1-difficulties]

        # Apply transformations
        # image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
           Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

           This describes how to combine these tensors of different sizes. We use lists.

           Note: this need not be defined in this Class, can be standalone.

           :param batch: an iterable of N sets from __getitem__()
           :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

if __name__ == '__main__':
    dataset = PascalVOCDataset('./', 'TRAIN')
    data = next(iter(dataset))
    print(data)




