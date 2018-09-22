import torch
import torch.nn.functional as F
from torch import nn
# from torchvision import models
from models import DRN


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out

BatchNorm = nn.BatchNorm2d

class DRN_GCN(nn.Module):
    def __init__(self, num_classes, layers=50,img_gray=False):
        super(DRN_GCN, self).__init__()
        resnet = DRN.drn_d_54(pretrained=True)
        resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        resnet.fc = nn.Conv2d(512, 2, kernel_size=1,
                             stride=1, padding=0, bias=True)
        resnet.load_state_dict(torch.load("/media/hszc/model/detao/models/salt/drn50_classfication/weights-171-250-[0.9425].pth"))
        resnet_layer0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                      bias=False),
            BatchNorm(16),
            nn.ReLU(inplace=True)
        ) if img_gray else resnet.layer0

        self.layer0 = nn.Sequential(resnet_layer0, resnet.layer1, resnet.layer2)
        self.layer1 = resnet.layer3
        self.layer2 = resnet.layer4
        self.layer3 = nn.Sequential(resnet.layer5, resnet.layer6)
        self.layer4 = nn.Sequential(resnet.layer7, resnet.layer8)

        self.gcm1 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        H,W = x.size()[2:]
        # if x: 512
        fm0 = self.layer0(x)  # 256
        # print(fm0.size())
        fm1 = self.layer1(fm0)  # 128
        # print(fm1.size())
        fm2 = self.layer2(fm1)  # 64
        # print(fm2.size())
        fm3 = self.layer3(fm2)  # 32
        # print(fm3.size())
        fm4 = self.layer4(fm3)  # 16
        # print(fm4.size())

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample(gcfm1, fm3.size()[2:],mode='bilinear') + gcfm2)  # 32
        fs2 = self.brm6(F.upsample(fs1, fm2.size()[2:],mode='bilinear') + gcfm3)  # 64
        fs3 = self.brm7(F.upsample(fs2, fm1.size()[2:],mode='bilinear') + gcfm4)  # 128
        fs4 = self.brm8(F.upsample(fs3, fm0.size()[2:],mode='bilinear'))  # 256
        out = self.brm9(F.upsample(fs4, (H,W),mode='bilinear'))  # 512
        return out

if __name__ =='__main__':
    model = DRN_GCN(2,  layers=50)
    # print model
    x = torch.FloatTensor(2,3,576,576)
    x = torch.autograd.Variable(x)
    y = model(x)
    print y.size()