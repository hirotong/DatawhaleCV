# !/usr/bin/env python
# !-*-coding:utf-8-*-
# @ Date: 2020/5/25 17:07
# @ Description:

import time
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.utils.data
from .model import SSD300, MultiBoxLoss
from .datasets import PascalVOCDataset
from .utils import *

cudnn.benchmark = True

# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = len(label_map)

# TODO
def main(config):
    """
    Training 
    """
    global label_map

    out_dir = './models'
    out_dir = os.path.join('./models', config.model_name)
    log = Logger()
    log.open(os.path.join(out_dir, config.model_name+'.txt'), mode='a')
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')

    # Initialize model or load checkpoint
    if config.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, which twice the default learning rate for bias
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[
            {'params': biases, 'lr': 2 * config.lr},
            {'params': not_biases}], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        log('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_dataset = PascalVOCDataset(data_dir=config.data_folder, split='train', keep_difficult=config.keep_difficult)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, num_workers=config.num_workers, pin_memory=True)

    for epoch in range(start_epoch, config.epochs):
        if epoch in config.decay_lr_at:
            adjust_learning_rate(optimizer, config.decay_lr_to)

        train


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training
    :param train_loader: DataLoader for training data
    :param model:   model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()

    bat



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSD object detection training")

    parser.add_argument('--model_name', type=str, default="SSD300-VOC")
    # parser.add_argument('--mode', type=str, default='train')
    # Data parameters
    parser.add_argument('--data_folder', type=str, default='./', help='path that contains json files.')
    parser.add_argument('--keep_difficult', type=bool, default=True)

    # Learning parameters
    parser.add_argument('--chcekpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay_lr_at', type=list, default=[150, 200])
    parser.add_argument('--decay_lr_to', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=None)

    config = parser.parse_args()
    print(config)
    main(config)