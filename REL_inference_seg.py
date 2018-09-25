import os
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm
from models.GCN import *
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from skimage.transform import resize
from dataset.data_aug import Compose,ResizeImg, Normalize
from torch.nn import functional as F
from  dataset.REL_dataset import REL_data,collate_fn_test
import torch.utils.data as torchdata
from models.drn_gcn import DRN_GCN
from dataset.data_aug import Cbct_crop
import argparse
import glob
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')

parser.add_argument('--cuda_device', type=str, default="0", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='gcn',help='gcn, drn_gcn,mobile_unet,deeplabv3')
parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/REL/gcn_512/weights-43-1480-[0.85046]-[0.96851].pth", help='path to inference weights file')
parser.add_argument('--debug', type=bool, default=True, help='where debug mode, plot image')
parser.add_argument('--mode', type=str, default='test', help='test,val')

parser.add_argument('--img_root_train', type=str, default= "/media/hszc/model/detao/data/nj_cbct/CBCT_trainingset/", help='whether to train img root')
parser.add_argument('--img_root_test', type=str, default= "/media/hszc/model/detao/data/nj_cbct/CBCT_testingset/test/", help='whether to test img root')
parser.add_argument('--img_root_result', type=str, default= "/media/hszc/model/detao/data/nj_cbct/CBCT_testingset/result", help='whether to test img root')

parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=386, help='size of each image dimension')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
class Test_Aug(object):
    def __init__(self,size=(opt.img_size,opt.img_size)):
        self.augment = Compose([
            Cbct_crop(x0=0, y0=200, size=512),
            Normalize(mean=None, std=None)
        ])
    def __call__(self, *args):
        return self.augment(*args)

# gcn, drn_gcn,mobile_unet
if opt.net == 'drn_gcn':
    model = DRN_GCN(4, layers=50)
elif opt.net == 'gcn':
    model = GCN(num_classes=4)
else:
    print("network is not defined")
if opt.resume:
    model.eval()
    print ('resuming finetune from %s' % opt.resume)
    try:
        model.load_state_dict(torch.load(opt.resume))
    except KeyError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(opt.resume))
model.cuda()
# val_path = "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_trainingset/Edema_trainingset/"
val_path = "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_validationset/Edema_validationset/"
val_pd = pd.DataFrame(glob.glob(os.path.join(val_path, "original_images") + "/*/*.bmp"), columns=['image_path'])
val_pd['id'] = val_pd["image_path"].apply(lambda x: x.split('/')[-1])
val_pd['label_path'] = val_pd["image_path"].apply(
    lambda x: x.replace("original_images", "label_images").replace("cube_z.img", "cube_z_labelMark"))
test_path="/media/hszc/model/detao/data/fl/ai_challenger_fl2018_testset/Edema_testset/original_images"
imglist =glob.glob(test_path+"/*/*.bmp")
# # imglist=imglist[:100]
test_pd=pd.DataFrame(imglist,columns=['image_path'])
test_pd=test_pd if opt.mode=="test" else val_pd
print(test_pd.head())

data_set = {}
data_loader = {}
img_root=opt.img_root_test if opt.mode=="test" else val_path

data_set['test'] = REL_data(img_root, list(test_pd['image_path'].values), Test_Aug(),test=True)
data_loader['test'] = torchdata.DataLoader(data_set['test'], opt.batch_size, num_workers=opt.n_cpu,
                                          shuffle=False, pin_memory=True, collate_fn=collate_fn_test)
def plot_cmp(image,mask,ori_img):
    f, axarr = plt.subplots(1,3)
    # axarr[0].imshow(image)

    axarr[0].imshow(image,cmap='seismic')
    axarr[0].imshow(mask,alpha=0.5)
    axarr[1].imshow(mask)
    axarr[2].imshow(ori_img,cmap='seismic')

    axarr[0].grid()
    axarr[1].grid()
    axarr[2].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')
    axarr[2].set_title('ori_img')
    plt.show()
for bc_data in tqdm(data_loader['test']):
    imgs, _,image_paths = bc_data
    imgs = Variable(imgs).cuda()

    output = model(imgs)
    output=F.softmax(output,dim=1)

    _, preds = torch.max(output, 1)  # (bs, H, W)
    preds=preds.cpu().data.numpy()
    preds[preds == 3] = 255
    preds[preds == 1] = 3
    preds[preds == 255] = 1
    for pred,img,image_path in zip(preds,imgs,image_paths):
        img = img.cpu().data.numpy().transpose((1,2,0))

        image_thr=img[:,:,0]*255
        if opt.mode=="val":
            mask_path =image_path.replace("original_images", "label_images").replace("cube_z.img","cube_z_labelMark")
            ori_img = cv2.imread(mask_path)
            plot_cmp(image_thr,pred,ori_img[:,:,0])
        save_path = image_path.replace("original_images", "result_images")
        if not os.path.exists(save_path.rsplit("/",1)[0]):
            os.makedirs(save_path.rsplit("/",1)[0])
        cv2.imwrite(save_path,pred)
