import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d, QLinear


def VGG8(args, logger):
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = nn.Sequential(
        QConv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv1_'),
        nn.ReLU(),
        QConv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv2_'),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        QConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv3_'),
        nn.ReLU(),
        QConv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv4_'),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        QConv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv5_'),
        nn.ReLU(),
        QConv2d(in_channels=channel[2], out_channels=channel[2], kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name='conv6_'),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        QLinear(in_features=16 * channel[2], out_features=fc_size, logger=logger,
                    wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                    wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target, name='fc1_'),
        nn.ReLU(),
        QLinear(in_features=fc_size, out_features=10, logger=logger,
                    wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                    wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target, name='fc2_')
    )
    return model