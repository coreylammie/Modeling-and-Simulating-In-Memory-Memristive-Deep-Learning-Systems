import argparse
import os
import time
from torch._C import device
from utee import misc
import torch
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from datetime import datetime
from utee import wage_quantizer
from utee import hook
import numpy as np
import random
import csv
from subprocess import call
from modules.quantization_cpu_np_infer import QConv2d, QLinear
import math
import ssl
import sys
import pandas as pd
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.getcwd() + "/..")

from LoadCIFAR import LoadCIFAR10
from GoogLeNet import GoogLeNet


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


# -------------------------------------------------------
neptune_project_name = ''
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
batch_size = 128
seed = 1
output_dir = 'output_Inference_GoogLeNet'
logdir = 'log'
weight_precision = 6
max_level = 64  # floor(log2(max_level)) = cellBit
r_on_off_ratio = 10
tile_shape = 16
ADC_precision = 6
nonlinearity_LTP = 1.75
nonlinearity_LTD = 1.46
d2d_variation = 0.1
c2c_variation = 0
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

args = {'batch_size': batch_size,
        'seed': seed,
        'logdir': logdir,
        'wl_weight': weight_precision,
        'max_level': max_level,
        'wl_grad': weight_precision,
        'wl_activate': weight_precision,
        'wl_error': weight_precision,
        'inference': 1,
        'onoffratio': r_on_off_ratio,
        'cellBit': weight_precision,
        'subArray': tile_shape,
        'ADCprecision': ADC_precision,
        'vari': 0,
        't': 0,
        'v': 0,
        'detect': 0,
        'target': 0,
        'nonlinearityLTP': nonlinearity_LTP,
        'nonlinearityLTD': nonlinearity_LTD,
        'max_level': 100,
        'vari': 0.1}

args = Struct(**args)
args.logdir = os.path.join(os.path.dirname(__file__), output_dir, args.logdir)
args = make_path.makepath(
    args, ['log_interval', 'test_interval', 'logdir', 'epochs', 'gpu', 'ngpu', 'debug'])
logger = misc.logger.info
misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

if __name__ == "__main__":
    args.cuda = device != torch.device("cpu")
    train_loader, _, test_loader = LoadCIFAR10(
        batch_size=batch_size, validation=False, num_workers=1)
    modelCF = GoogLeNet(args=args, logger=logger).to(device)
    best_acc, old_file = 0, None
    modelCF.eval()
    df = pd.DataFrame(columns=['idx', 'elapsed_time',
                      'test_set_accuracy', 'test_loss'])
    run = neptune.init(
        project=neptune_project_name,
        api_token=neptune_api_token,
    )
    with torch.no_grad():
        for iteration in range(10):
            test_loss = 0
            correct = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i, (data, target) in enumerate(test_loader):
                if i == 0:
                    hook_handle_list = hook.hardware_evaluation(
                        modelCF, args.wl_weight, args.wl_activate, 0)

                indx_target = target.clone()
                if args.cuda:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                output = modelCF(data)
                test_loss += torch.functional.cross_entropy(
                    output, target).data
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(indx_target).sum()
                if i == 0:
                    hook.remove_hook_list(hook_handle_list)

            test_loss = test_loss / len(test_loader)
            acc = 100. * correct / len(test_loader.dataset)
            accuracy = acc.cpu().data.numpy()
            end.record()
            torch.cuda.synchronize()
            df = df.append({'idx': iteration, 'elapsed_time': start.elapsed_time(
                end), 'test_set_accuracy': accuracy.item(), 'test_loss': test_loss.cpu().item()}, ignore_index=True)
            df.to_csv('Inference_GoogLeNet.csv', index=False)

    run.stop()
    print(df)
