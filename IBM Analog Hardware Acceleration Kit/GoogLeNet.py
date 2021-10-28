import torch
import torch.nn as nn
import aihwkit
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential


class Inception(nn.Module):
    def __init__(self, RPU_CONFIG, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        self.b1 = AnalogSequential(
            AnalogConv2d(in_planes, kernel_1_x, kernel_size=1,
                         bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(),
        )
        self.b2 = AnalogSequential(
            AnalogConv2d(in_planes, kernel_3_in, kernel_size=1,
                         bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(),
            AnalogConv2d(kernel_3_in, kernel_3_x, kernel_size=3,
                         padding=1, bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(),
        )
        self.b3 = AnalogSequential(
            AnalogConv2d(in_planes, kernel_5_in, kernel_size=1,
                         bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(),
            AnalogConv2d(kernel_5_in, kernel_5_x, kernel_size=3,
                         padding=1, bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
            AnalogConv2d(kernel_5_x, kernel_5_x, kernel_size=3,
                         padding=1, bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(),
        )
        self.b4 = AnalogSequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            AnalogConv2d(in_planes, pool_planes, kernel_size=1,
                         bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, RPU_CONFIG, n_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = AnalogSequential(
            AnalogConv2d(3, 192, kernel_size=3, padding=1,
                         bias=True, rpu_config=RPU_CONFIG),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.a3 = Inception(RPU_CONFIG, 192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(RPU_CONFIG, 256, 128, 128, 192, 32, 96, 64)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(RPU_CONFIG, 480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(RPU_CONFIG, 512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(RPU_CONFIG, 512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(RPU_CONFIG, 512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(RPU_CONFIG, 528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(RPU_CONFIG, 832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(RPU_CONFIG, 832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = AnalogLinear(
            1024, n_classes, bias=True, rpu_config=RPU_CONFIG)

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
