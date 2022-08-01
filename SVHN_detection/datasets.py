# !/usr/bin/env python
# !-*-coding:utf-8-*-

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import transform


class SVHNDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.split = split.lower()
        assert self.split in {'train', 'val', 'test'}

        self.data_folder = os.path.abspath(os.path.join(data_dir, f'mchar_{self.split}'))

        with open(os.path.join(data_dir, f'mchar_{self.split}.json'), 'r') as j:
            self.objects = json.load(j)
        self.images = list(self.objects)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.data_folder,self.images[index]), mode='r')
        image = image.convert('RGB')

        boxes = []
        labels = []
        if self.split != 'test':
            object = self.objects[self.images[index]]
            boxes = torch.FloatTensor(list(zip(object['left'], object['top'], object['width'], object['height'])))
            boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], dim=1)
            labels = torch.LongTensor(object['label']) + 1

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels

    def collate_fn(self, batch):
        """
        Since the images, boxes, labels may be in different sizes. The collate_fn is used to pass to DataLoader.
        :param batch: one batch return by dataset
        :return: image: torch.FloatTensor, boxes: list, labels: list
        """
        images = []
        boxes = []
        labels = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)   #(N, 3, 300, 300)

        return images, boxes, labels





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
        with open(os.path.join(data_dir, f'{self.split}_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_dir, f'{self.split}_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, index):
        image = Image.open(self.images[index], mode='r')
        image = image.convert('RGB')

        objects = self.objects[index]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.FloatTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

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
        images = []
        boxes = []
        labels = []
        difficulties = []
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


if __name__ == '__main__':
    dataset = SVHNDataset('../Dataset', 'train')
    data = next(iter(dataset))
    print(data)
    data[0].show()
