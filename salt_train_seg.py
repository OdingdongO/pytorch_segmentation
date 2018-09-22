from utils.seg_salt_train_util import train, trainlog
from torch.optim import lr_scheduler,Adam,RMSprop
from models.GCN import *
from utils.seg_losses import CrossEntropyLoss2d
from dataset.data_aug import Compose,ResizeImg,RandomHflip,Normalize,RandomVflip
from models.drn_gcn import DRN_GCN
from models.drn_gcn_depths_salt import DRN_GCN_Depth
import os
import pandas as pd
from  dataset.salt_seg_dataset import Salt_data,collate_fn
from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('--learning_rate', type=float, default=1e-03, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/salt/drn_gcn_Adam_gray_bs8_resume', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="2", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='drn_gcn',help='gcn, drn_gcn,mobile_unet,deeplabv3')
parser.add_argument('--with_depth', type=bool, default=False, help='whether train model with depth')
parser.add_argument('--optim', dest='optim',type=str, default='Adam',help='Adam,RMSprop')
parser.add_argument('--img_gray', dest='img_gray',type=bool, default=True,help='img gray')
# /media/hszc/model/detao/models/salt/drn_gcn_Adam_gray/bestweights-[0.881].pth
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')

parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=400, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
parser.add_argument('--list_path', type=str, default='/media/hszc/model/detao/data/salt/train.csv', help='whether to data anno')
parser.add_argument('--depth_path', type=str, default='/media/hszc/model/detao/data/salt/depths.csv', help='whether to depth data anno')
parser.add_argument('--img_root', type=str, default= "/media/hszc/model/detao/data/salt/train/", help='whether to img root')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

class train_Aug(object):
    def __init__(self, size=(opt.img_size, opt.img_size)):
        self.augment = Compose([
            # RandomSelect([
            #     RandomSmall(ratio=0.1),
            #     RandomRotate(angles=(-20, 20), bound='Random'),
            #     RandomResizedCrop(size=size),
            # ]),
            # RandomBrightness(delta=30),
            ResizeImg(size=size),
            RandomHflip(),
            RandomVflip(),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

class val_Aug(object):
    def __init__(self,size=(opt.img_size,opt.img_size)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

def lr_lambda(epoch):
    if epoch < 50:
        return 1
    elif epoch < 75:
        return 0.1
    elif epoch < 90:
        return 0.05
    else:
        return 0.01

if __name__ == '__main__':
    depths_df = pd.read_csv(opt.depth_path)
    depths_df['z']=depths_df['z']/1000

    image_df = pd.read_csv(opt.list_path)
    image_df = pd.merge(image_df, depths_df, how="left", on='id')

    train_pd, val_pd = train_test_split(image_df, test_size=0.1, random_state=34)

    # saving dir
    save_dir = opt.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = '%s/trainlog.log' % save_dir
    trainlog(logfile)

    data_set = {}
    data_loader = {}

    data_set['train'] = Salt_data(opt.img_root, train_pd, train_Aug(),gray=opt.img_gray)
    data_loader['train'] = torchdata.DataLoader(data_set['train'], opt.batch_size, num_workers=opt.n_cpu,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
    data_set['val'] = Salt_data(opt.img_root, val_pd, val_Aug(),gray=opt.img_gray)
    data_loader['val'] = torchdata.DataLoader(data_set['val'], 4, num_workers=opt.n_cpu,
                                              shuffle=False, pin_memory=True, collate_fn=collate_fn)

    print len(data_set['train']), len(data_set['val'])

    # gcn, drn_gcn,mobile_unet
    if opt.net == 'drn_gcn':
        model = DRN_GCN(2, layers=50,img_gray=opt.img_gray) if not opt.with_depth else DRN_GCN_Depth(2, layers=50,img_gray=opt.img_gray)
    elif opt.net == 'gcn':
        model = GCN(num_classes=2)
    else:
        print("network is not defined")
    logging.info(model)

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
        # optimizer.load_state_dict(torch.load(os.path.join(opt.resume, 'optimizer-state.pth')))
    model.cuda()

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
          with_depth=opt.with_depth
          )