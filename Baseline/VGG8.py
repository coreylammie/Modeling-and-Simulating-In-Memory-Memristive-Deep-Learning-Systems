import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG8(nn.Module):
    def __init__(self):
        super(VGG8, self).__init__()
        channel_base = 48
        channel = [channel_base, 2 * channel_base, 3 * channel_base]
        fc_size = 8 * channel_base
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[2], out_channels=channel[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Flatten(),
            nn.Linear(in_features=16 * channel[2], out_features=fc_size, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=fc_size, out_features=10, bias=True)
        )
    def forward(self, x):
        return self.model(x)