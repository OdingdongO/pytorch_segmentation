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
    def __init__(self, root_path, anno_pd,transforms=None,debug=False,test=False,gray=True):
        self.root_path = root_path
        self.file_list = list(anno_pd['id'].values)
        self.depth_list = list(anno_pd['z'].values)
        self.transforms = transforms
        self.debug =debug
        self.test =test
        self.gray = gray

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        file_id = self.file_list[index]
        depth = self.depth_list[index]
        image_path = os.path.join(self.root_path, "images", file_id + ".png")
        if not self.gray:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        else:
            img=cv2.imread(image_path,0)

        if not self.test:
            mask_path = os.path.join(self.root_path, "masks", file_id + ".png")
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            mask[mask<255] = 0
            mask[mask==255] = 1
        else:
            mask = np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)
        mask_teacher=None
        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)
        label = 1 if np.sum(mask)>0.5 else 0
        if self.gray:
            img = np.expand_dims(img, 2)
        if not self.debug:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        if not self.test:
            return img, mask,depth
        else:
            return img, mask,file_id,depth

def collate_fn_test(batch):
    imgs = []
    masks = []
    file_id = []
    depth = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        file_id.append(sample[2])
        depth.append(sample[3])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0), \
           file_id,depth
def collate_fn(batch):
    imgs = []
    masks = []
    depth = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        depth.append(sample[2])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0),\
            depth
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,3)
    img_cumsum = (np.float32(image) - image.mean()).cumsum(axis=0)
    axarr[0].imshow(image,cmap='seismic')
    axarr[1].imshow(img_cumsum*img_cumsum, cmap='seismic')
    axarr[2].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[2].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Cumsum')
    axarr[2].set_title('Mask')

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from dataset.data_aug import *
    from utils.salt_submission  import RLenc
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
                # Normalize(mean=None, std=None)
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
    image_gray=True
    train_path = "/media/hszc/model/detao/data/salt/train/"
    # file_list = list(depths_df['id'].values)
    depths_df = pd.read_csv('/media/hszc/model/detao/data/salt/depths.csv')
    depths_df['z']=depths_df['z']/1000
    # depths_df.hist()
    # plt.show()
    defect_dict = dict(depths_df['z'].value_counts())
    print(len(defect_dict))
    print(depths_df.head(),depths_df.shape)
    image_df = pd.read_csv('/media/hszc/model/detao/data/salt/train.csv')
    image_df=pd.merge(image_df,depths_df,how="left",on='id')
    print image_df.head()
    print(image_df.shape)
    train_pd, val_pd = train_test_split(image_df, test_size=0.1, random_state=34)
    print(train_pd.shape,val_pd.shape,image_df.shape)
    dataset = Salt_data(train_path, train_pd,trainAug(),debug=True,gray=image_gray)

    # dataset = Salt_data(train_path, file_list)
    for data in dataset:

    # for i in range(5):
    #     image, mask = dataset[np.random.randint(0, len(dataset))]
        image, mask,depth =data
        print(image.shape)

        value =RLenc(mask)
        plot2x2Array(image[:, :, 0], mask)

        plt.show()