# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

class Salt_data(data.Dataset):
    def __init__(self, root_path, file_list,transforms=None,debug=False,test=False):
        self.root_path = root_path
        self.file_list = file_list
        self.transforms = transforms
        self.debug =debug
        self.test =test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        file_id = self.file_list[index]
        image_path = os.path.join(self.root_path, "images", file_id + ".png")
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        if not self.test:
            mask_path = os.path.join(self.root_path, "masks", file_id + ".png")
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            mask[mask<255] = 0
            mask[mask==255] = 1
            label = 1 if np.sum(mask) > 0.5 else 0
        else:
            mask = np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)
            label=None
        mask_teacher=None
        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)

        if not self.debug:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        if not self.test:
            return img, mask,label
        else:
            return img, mask,file_id

def collate_fn_test(batch):
    imgs = []
    masks = []
    file_id = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        file_id.append(sample[2])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0), \
           file_id
def collate_fn(batch):
    imgs = []
    masks = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0)
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from dataset.data_aug import *
    from utils.submission import RLenc
    class trainAug(object):
        def __init__(self, size=(101, 101)):
            self.augment = Compose([
                # RandomSelect([
                #     RandomSmall(ratio=0.1),
                #     RandomRotate(angles=(-20, 20), bound='Random'),
                #     RandomResizedCrop(size=size),
                # ]),
                # RandomBrightness(delta=30),
                ResizeImg(size=size),
                # RandomHflip(),
                Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)


    class valAug(object):
        def __init__(self, size=(101, 101)):
            self.augment = Compose([
                ResizeImg(size=size),
                Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)
    train_path = "/media/hszc/model/detao/data/salt/train/"
    # file_list = list(depths_df['id'].values)
    depths_df = pd.read_csv('/media/hszc/model/detao/data/salt/train.csv')
    print depths_df.head()
    train_pd, val_pd = train_test_split(depths_df, test_size=0.1, random_state=34)
    print(train_pd.shape,val_pd.shape,depths_df.shape)
    dataset = Salt_data(train_path, list(train_pd['id'].values),trainAug(),debug=True)

    # dataset = Salt_data(train_path, file_list)
    for data in dataset:

    # for i in range(5):
    #     image, mask = dataset[np.random.randint(0, len(dataset))]
        image, mask =data
        value =RLenc(mask)
        print(value)
        plot2x2Array(image, mask)
        plt.show()