from __future__ import division
import os,time,datetime
import logging
from math import ceil
from copy import deepcopy
from logs import *
# from utils.preprocessing import *
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.plotting import getPlotImg
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from metrics import cal_IOU

def predict(model, data_set, data_loader, counting=False,with_depth=False):
    """ Validate after training an epoch
    Note:
    """
    model.eval()
    n = len(data_set)
    ious = np.zeros(n,dtype=float)
    idx = 0
    val_acc =[]
    for bc_cnt, bc_data in enumerate(data_loader):
        if counting:
            print('%d/%d' % (bc_cnt, len(data_set)//data_loader.batch_size))
        imgs, masks,depth = bc_data
        imgs = Variable(imgs).cuda()
        masks = Variable(masks).cuda()
        depth = Variable(torch.from_numpy(np.array(depth)).float().cuda())
        outputs = model(imgs) if not with_depth else model(imgs, depth)
        outputs = F.softmax(outputs, dim=1)
        if outputs.size() != masks.size():
            outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')


        # cal pixel acc
        _, outputs = torch.max(outputs, dim=1)
        batch_corrects = torch.sum((outputs == masks).long()).data[0]
        batch_acc = 1. * batch_corrects / (masks.size(0) * masks.size(1) * masks.size(2))
        val_acc.append(batch_acc)
        outputs = outputs.cpu().data.numpy()
        masks = masks.cpu().data.numpy()
        ious[idx: idx+imgs.size(0)] = cal_IOU(outputs, masks,2)
        idx += imgs.size(0)

    return ious.mean(),np.array(val_acc).mean()

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500,
          with_depth=False,
          ):

    writer = SummaryWriter(save_dir)
    best_model_wts = model.state_dict()
    best_mIOU = 0

    running_loss = 20
    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode


        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            model.train(True)
            imgs, masks,depth= data

            imgs = Variable(imgs.cuda())
            masks = Variable(masks.cuda())
            depth = Variable(torch.from_numpy(np.array(depth)).float().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(imgs) if not with_depth else model(imgs,depth)

            # print outputs.size(), masks.size()
            if outputs.size() != masks.size():
                outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')

            # print outputs.size()
            # print masks.size()
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            running_loss = running_loss*0.95 + 0.05*loss.data[0]

            # cal pixel acc
            _, preds = torch.max(outputs,1)  # (bs, H, W)
            # preds = F.softmax(outputs,dim=1).round()[:, 1, :].long()
            batch_corrects = torch.sum((preds==masks).long()).data[0]
            batch_acc = 1.*batch_corrects / (masks.size(0)*masks.size(1)*masks.size(2))

            iou = cal_IOU(preds.data.cpu().numpy(), masks.cpu().data.numpy(), 2)

            if step % print_inter == 0:
                logging.info('%s [%d-%d] | loss: %.3f | run-loss: %.3f | acc: %.3f | miou: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data[0], running_loss, batch_acc, iou.mean()))

            # # plot image
            # if step % (2*print_inter) == 0:
            #     smp_img = imgs[0]  # (3, H, W)
            #     true_hm = masks[0]  #(H,W)
            #     pred_hm = F.softmax(outputs[0])[1]
            #
            #     imgs_to_plot = getPlotImg(smp_img, pred_hm, true_hm)
            #
            #     # for TensorBoard
            #     imgs_to_plot = torch.from_numpy(imgs_to_plot.transpose((0,3,1,2))/255.0)
            #     grid_image = make_grid(imgs_to_plot, 2)
            #     writer.add_image('plotting',grid_image, step)
            #     writer.add_scalar('loss', loss.data[0],step)

            if step % val_inter == 0:
                # val phase
                model.eval()

                t0 = time.time()
                mIOU,val_acc = predict(model, data_set['val'], data_loader['val'], counting=False,with_depth=with_depth)
                t1 = time.time()
                since = t1-t0

                logging.info('--' * 30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d] | val-mIOU@1: %.4f%% | val-acc: %.3f | time: %d'
                             % (dt(), epoch, mIOU,val_acc, since))

                if mIOU > best_mIOU:
                    best_mIOU = mIOU
                    best_model_wts = deepcopy(model.state_dict())

                # save model
                save_path1 = os.path.join(save_dir,
                        'weights-%d-%d-[%.4f]-[%.4f].pth'%(epoch,batch_cnt,mIOU,val_acc))
                torch.save(model.state_dict(), save_path1)
                save_path2 = os.path.join(save_dir,
                        'optimizer-state.pth')
                torch.save(optimizer.state_dict(), save_path2)

                logging.info('saved model to %s' % (save_path1))
                logging.info('--' * 30)

    # save best model
    save_path = os.path.join(save_dir,
                             'bestweights-[%.3f].pth' % (best_mIOU))
    torch.save(best_model_wts, save_path)
    logging.info('saved model to %s' % (save_path))

    return best_mIOU, best_model_wts