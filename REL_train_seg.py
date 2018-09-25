from utils.seg_train_util import train, trainlog
from torch.optim import lr_scheduler,Adam,RMSprop
from models.GCN import *
from utils.seg_losses import CrossEntropyLoss2d
from dataset.data_aug import Compose,ResizeImg,RandomHflip,Normalize
from models.drn_gcn import DRN_GCN
import os
import pandas as pd
from  dataset.REL_dataset import REL_data,collate_fn
from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
import logging
import argparse
from dataset.data_aug import UpperCrop,RandomUpperCrop
from dataset.data_aug import Cbct_crop, Cbct_random_crop,Hflip
import glob
'''
https://challenger.ai/competition/fl2018
'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4, help='size of each image batch')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/REL/gcn_512', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="2,3", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='gcn',help='gcn, drn_gcn,mobile_unet,deeplabv3')
parser.add_argument('--optim', dest='optim',type=str, default='Adam',help='Adam,RMSprop')
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')

parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=600, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
parser.add_argument('--img_root_train', type=str, default= "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_trainingset/Edema_trainingset/", help='whether to img root')
parser.add_argument('--img_root_val', type=str, default= "/media/hszc/model/detao/data/fl/ai_challenger_fl2018_validationset/Edema_validationset/", help='whether to img root')

parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=512, help='size of each image dimension')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

class train_Aug(object):
    def __init__(self, size=(opt.img_size, opt.img_size)):
        self.augment = Compose([
            Cbct_crop(x0=0, y0=200, size=512),
            RandomHflip(),
            Normalize(mean=None, std=None)

        ])

    def __call__(self, *args):
        return self.augment(*args)

class val_Aug(object):
    def __init__(self,size=(opt.img_size,opt.img_size)):
        self.augment = Compose([
            Cbct_crop(x0=0, y0=200, size=512),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

def lr_lambda(epoch):
    if epoch < 45:
        return 1
    elif epoch < 75:
        return 0.1
    elif epoch < 90:
        return 0.05
    else:
        return 0.01

if __name__ == '__main__':

    train_pd =pd.DataFrame(glob.glob(os.path.join(opt.img_root_train,"original_images")+"/*/*.bmp"),columns=['image_path'])
    train_pd['id'] =train_pd["image_path"].apply(lambda x:x.split('/')[-1])
    train_pd['label_path'] =train_pd["image_path"].apply(lambda x:x.replace("original_images","label_images").replace("cube_z.img","cube_z_labelMark"))

    val_pd =pd.DataFrame(glob.glob(os.path.join(opt.img_root_val,"original_images")+"/*/*.bmp"),columns=['image_path'])
    val_pd['id'] =val_pd["image_path"].apply(lambda x:x.split('/')[-1])
    val_pd['label_path'] =val_pd["image_path"].apply(lambda x:x.replace("original_images","label_images").replace("cube_z.img","cube_z_labelMark"))

    # saving dir
    save_dir = opt.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = '%s/trainlog.log' % save_dir
    trainlog(logfile)

    data_set = {}
    data_loader = {}

    data_set['train'] = REL_data(train_path, list(train_pd['image_path'].values), train_Aug())
    data_loader['train'] = torchdata.DataLoader(data_set['train'], opt.batch_size, num_workers=opt.n_cpu,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
    data_set['val'] = REL_data(val_path, list(val_pd['image_path'].values), val_Aug())
    data_loader['val'] = torchdata.DataLoader(data_set['val'], 3, num_workers=opt.n_cpu,
                                              shuffle=False, pin_memory=True, collate_fn=collate_fn)

    print len(data_set['train']), len(data_set['val'])

    # gcn, drn_gcn,mobile_unet
    if opt.net == 'drn_gcn':
        model = DRN_GCN(4, layers=50)
    elif opt.net == 'gcn':
        model = GCN(num_classes=4)
    else:
        print("network is not defined")
    # logging.info(model)

    criterion = CrossEntropyLoss2d()

    if opt.optim == 'Adam':
        optimizer = Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    elif opt.optim == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if opt.resume:
        model.eval()
        logging.info('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
    model.cuda()
    model=torch.nn.DataParallel(model)

    train(model,
          epoch_num=opt.epochs,
          start_epoch=opt.start_epoch,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=data_loader,
          save_dir=save_dir,
          print_inter=opt.print_interval,
          val_inter=opt.save_checkpoint_val_interval,
          )