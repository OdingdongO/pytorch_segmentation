# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

class REL_data(data.Dataset):
    def __init__(self, root_path, file_list,transforms=None,debug=False,test=False):
        self.root_path = root_path
        self.file_list = file_list
        self.transforms = transforms
        self.debug =debug
        self.test =test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        image_path = self.file_list[index]
        # file_id =image_path.split("/")[-1]
        if not self.test:
            mask_path =image_path.replace("original_images", "label_images").replace("cube_z.img","cube_z_labelMark")
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            mask[mask==128] = 1
            mask[mask==191] = 2
            mask[mask==255] = 3
            mask[mask>3] = 0

        else:
            # image_path = os.path.join(self.root_path, file_id)
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
            mask = np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)
            # if file_id.split('+')[0]=="04":
            #     img[img < 100] = 0
            # else:
            #     img[img < 100] = 0
                # img=img*0.95
        mask_teacher=None
        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)
        if not self.debug:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        if not self.test:
            return img, mask
        else:
            return img, mask,image_path

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
    f, axarr = plt.subplots(1,3)
    # axarr[0].imshow(image)
    axarr[0].imshow(image,cmap='seismic')
    axarr[0].imshow(mask,alpha=0.3)

    axarr[1].imshow(mask)
    axarr[2].imshow(image,cmap='seismic')

    axarr[0].grid()
    axarr[1].grid()
    axarr[2].grid()
    axarr[0].set_title('Image &Mask')
    axarr[1].set_title('Mask')
    axarr[2].set_title('Image')
    plt.show()

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from dataset.data_aug import *
    from  dataset.data_aug import DownCrop,Center_Crop,UpperCrop,RandomUpperCrop
    from dataset.data_aug import Cbct_crop,Cbct_random_crop

    class trainAug(object):
        def __init__(self, size=(386, 386)):
            self.augment = Compose([
                # RandomSelect([
                #     RandomSmall(ratio=0.1),
                #     RandomRotate(angles=(-20, 20), bound='Random'),
                #     RandomResizedCrop(size=size),
                # ]),
                # RandomBrightness(delta=30),
                # ResizeImg(size=size),
                # RandomUpperCrop(size=size),
                Cbct_crop(x0=0, y0=200, size=512),
                # RandomHflip(),
                # Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)


    class valAug(object):
        def __init__(self, size=(576, 576)):
            self.augment = Compose([
                # ResizeImg(size=size),
                Cbct_crop(x0=200, y0=0, size=512),
                # Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)

    train_path = "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_trainingset/Edema_trainingset/"
    val_path = "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_validationset/Edema_validationset/"
    # label_images
    # file_list = list(depths_df['id'].values)
    import glob
    train_pd =pd.DataFrame(glob.glob(os.path.join(train_path,"original_images")+"/*/*.bmp"),columns=['image_path'])
    train_pd['id'] =train_pd["image_path"].apply(lambda x:x.split('/')[-1])
    train_pd['label_path'] =train_pd["image_path"].apply(lambda x:x.replace("original_images","label_images").replace("cube_z.img","cube_z_labelMark"))

    val_pd =pd.DataFrame(glob.glob(os.path.join(val_path,"original_images")+"/*/*.bmp"),columns=['image_path'])
    val_pd['id'] =val_pd["image_path"].apply(lambda x:x.split('/')[-1])
    val_pd['label_path'] =val_pd["image_path"].apply(lambda x:x.replace("original_images","label_images").replace("cube_z.img","cube_z_labelMark"))

    print(train_pd.shape,val_pd.shape)

    dataset = REL_data(train_path,list(train_pd['image_path'].values),trainAug(),debug=True)

    # dataset = Salt_data(train_path, file_list)
    for data in dataset:

    # for i in range(5):
    #     image, mask = dataset[np.random.randint(0, len(dataset))]
        image, mask =data
        # image_thr=image[:, :, 0]*255
        # image_thr[image_thr < 1] = 255
        #
        # print(np.sort(image_thr,axis=None))
        # # image_thr = image[:, :, 0] * 255
        # image_thr[image_thr < 100] = 0
        # plot2x2Array(image[:,:,0], mask*255)
        plot2x2Array(image, mask)

        plt.show()