import torch
import torch.nn as nn
import aihwkit
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential


def VGG8(RPU_CONFIG):
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1, rpu_config=RPU_CONFIG),
        nn.ReLU(),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, stride=1,
                     padding=1,
                     rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=1,
                     padding=1,
                     rpu_config=RPU_CONFIG),
        nn.ReLU(),
        AnalogConv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, stride=1,
                     padding=1,
                     rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, stride=1,
                     padding=1,
                     rpu_config=RPU_CONFIG),
        nn.ReLU(),
        AnalogConv2d(in_channels=channel[2], out_channels=channel[2], kernel_size=3, stride=1,
                     padding=1,
                     rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        AnalogLinear(in_features=16 * channel[2], out_features=fc_size,
                     rpu_config=RPU_CONFIG),
        nn.ReLU(),
        AnalogLinear(in_features=fc_size, out_features=10,
                     rpu_config=RPU_CONFIG)
    )
    return model
