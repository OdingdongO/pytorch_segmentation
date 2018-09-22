import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, y_preds, y_true):
        '''
        :param y_preds: (N, C, H, W), Variable of FloatTensor
        :param y_true:  (N, H, W), Variable of LongTensor
        # :param weights: sample weights, (N, H, W), Variable of FloatTensor
        :return:
        '''
        logp = F.log_softmax(y_preds,dim=1)    # (N, C, H, W)
        ylogp = torch.gather(logp, 1, y_true.view(y_true.size(0), 1, y_true.size(1), y_true.size(2))) # (N, 1, H, W)
        return -(ylogp.squeeze(1)).mean()
