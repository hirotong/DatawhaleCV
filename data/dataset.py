'''
@Description: SVHN dataset
@Author: hiro.tong
@Date: 2020-05-19 09:10:16
@LastEditTime: 2020-05-19 18:31:55
'''
#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import json

ROOT_PATH=os.getcwd()


def load_img_list(file_path):
    img_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = f.readline().strip()
            img_list.append(line)
    return img_list


class SVHNDataset(Dataset):
    def __init__(self, mode,  data_dir, img_size=(28, 28),transform=None):
        super(SVHNDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.img_size = img_size

        self.img_list = load_img_list(os.path.join(self.data_dir, self.mode + '.txt'))
        if self.mode != 'test':
            self.labels = json.load(open(os.path.join(data_dir, 'mchar_' + self.mode + '.json')))

        
    def __getitem__(self, index):
        img_name = os.path.split(self.img_list[index])[1]

        pil_image = Image.open(os.path.join(ROOT_PATH, self.img_list[index])).convert('RGB').resize(self.img_size)
        
        h, w = pil_image.size
        
        img = self.transform(pil_image)
        
        if self.mode != 'test': 
            label_dict = self.labels[img_name]

            label = np.vstack((label_dict['label'], label_dict['left'], label_dict['top'], label_dict['width'], label_dict['height'])).transpose(1, 0)

            if label.shape[0] < 5:
                for i in range(5 - label.shape[0]):
                    label = np.concatenate((label, np.ones((1, 5)) * (-1)), axis=0)
        
            label = torch.tensor(label, dtype=torch.float32)

            sample = {
            "image": img,
            "label": label
            }
        else:
            sample = {
                "image": img
            }

        return sample

    def __len__(self):
        return len(self.img_list)
        

        

def load_data_SVHN(batch_size, img_size, transform = None,  data_dir='./Dataset', num_workers=1):
    
    train_dataset = SVHNDataset('train',  data_dir,img_size=img_size, transform=transform)
    val_dataset = SVHNDataset('val',  data_dir, img_size=img_size,transform=transform)
    test_dataset = SVHNDataset('test',  data_dir,img_size=img_size, transform=transform)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_iter = DataLoader(val_dataset, shuffle=False, num_workers=num_workers)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_iter, val_iter, test_iter


if __name__ == "__main__":
    train, val, test = load_data_SVHN(20, (28, 28), None, data_dir='./Dataset', num_workers=2)
    batch = iter(train).next()
    print(batch['label'])

    torchvision.models.detection.fasterrcnn_resnet50_fpn