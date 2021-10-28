import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d, QLinear

global inception_idx
inception_idx = 0

class Inception(nn.Module):
    def __init__(self, args, logger, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        global inception_idx
        self.b1 = nn.Sequential(
            QConv2d(in_planes, kernel_1_x, kernel_size=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv1_' % inception_idx),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            QConv2d(in_planes, kernel_3_in, kernel_size=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv2_' % inception_idx),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(),
            QConv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv3_' % inception_idx),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            QConv2d(in_planes, kernel_5_in, kernel_size=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv4_' % inception_idx),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(),
            QConv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv5_' % inception_idx),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
            QConv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv6_' % inception_idx),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            QConv2d(in_planes, pool_planes, kernel_size=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='inception_%d_conv7_' % inception_idx),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )
        inception_idx = inception_idx + 1

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, args, logger, n_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            QConv2d(3, 192, kernel_size=3, padding=1, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv1_'),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.a3 = Inception(args, logger, 192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(args, logger, 256, 128, 128, 192, 32, 96, 64)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(args, logger, 480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(args, logger, 512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(args, logger, 512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(args, logger, 512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(args, logger, 528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(args, logger, 832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(args, logger, 832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = QLinear(1024, n_classes, bias=True, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='fc1_')

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x