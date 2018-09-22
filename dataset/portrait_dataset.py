# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

class Sdata(data.Dataset):
    def __init__(self, anno_pd, transforms, dis=False):
        anno_pd.index = range(anno_pd.shape[0])
        self.image_paths = anno_pd['image_paths'].tolist()
        self.mask_paths = anno_pd['mask_paths'].tolist()
        self.mask_teacher_paths = anno_pd['mask_teacher_paths'].tolist()
        self.transforms = transforms
        self.dis = dis
        # deal with label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.image_paths[item]), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        mask = cv2.imread(self.mask_paths[item], cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        if self.dis:
            mask_teacher = np.load(self.mask_teacher_paths[item])
            mask_teacher = cv2.resize(mask_teacher, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_teacher = None

        mask[mask==2] = 1

        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        if not self.dis:
            return img, mask
        else:
            mask_teacher = torch.from_numpy(mask_teacher).float()
            return img, mask,\
                    mask_teacher


def collate_fn(batch):
    imgs = []
    masks = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0)

def collate_fn2(batch):
    imgs = []
    masks = []
    masks_teacher = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        masks_teacher.append(sample[2])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0), \
           torch.stack(masks_teacher, 0)


if __name__ == '__main__':
    from utils.preprocessing import get_train_val
    from Sdata.Saug import *


    class valAug(object):
        def __init__(self, size=(448, 448)):
            self.augment = Compose([
                ResizeImg(size=size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __call__(self, *args):
            return self.augment(*args)


    train_root = '/media/hszc/data1/seg_data'
    val_root = '/media/hszc/data1/seg_data/diy_seg'
    train_pd, _ = get_train_val(train_root, test_size=0.0)
    _, val_pd = get_train_val(val_root, test_size=1.0)
    print train_pd.info()
    print val_pd.info()

    data_set = Sdata(val_pd, valAug())

    print data_set[0][0].shape
    print np.unique(data_set[0][1])

