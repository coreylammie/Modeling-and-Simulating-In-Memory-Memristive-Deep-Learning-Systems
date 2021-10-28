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

from VGG8 import VGG8
from LoadCIFAR import LoadCIFAR10


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


# -------------------------------------------------------
neptune_project_name = ''
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
batch_size = 128
epochs = 256
seed = 1
log_interval = 100
test_interval = 1
output_dir = 'output_Train_VGG8'
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
initial_learning_rate = 0.1
momentum = 0.9
learning_rate_schedule = [100, 200, 250]
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


args = {'batch_size': batch_size,
        'epochs': epochs,
        'seed': seed,
        'log_interval': log_interval,
        'test_interval': test_interval,
        'logdir': logdir,
        'wl_weight': weight_precision,
        'max_level': max_level,
        'wl_grad': weight_precision,
        'wl_activate': weight_precision,
        'wl_error': weight_precision,
        'inference': 0,
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
        'd2dVari': d2d_variation,
        'c2cVari': c2c_variation}

args = Struct(**args)
gamma = initial_learning_rate
alpha = momentum
grad_scale = 1
decreasing_lr = learning_rate_schedule
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
NeuroSim_Out = np.array([["L_forward (s)", "L_activation gradient (s)", "L_weight gradient (s)", "L_weight update (s)",
                          "E_forward (J)", "E_activation gradient (J)", "E_weight gradient (J)", "E_weight update (J)",
                          "L_forward_Peak (s)", "L_activation gradient_Peak (s)", "L_weight gradient_Peak (s)", "L_weight update_Peak (s)",
                          "E_forward_Peak (J)", "E_activation gradient_Peak (J)", "E_weight gradient_Peak (J)", "E_weight update_Peak (J)",
                          "TOPS/W", "TOPS", "Peak TOPS/W", "Peak TOPS"]])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.savetxt(os.path.join(output_dir, "NeuroSim_Output.csv"),
           NeuroSim_Out, delimiter=",", fmt='%s')
if not os.path.exists(os.path.join(output_dir, 'NeuroSim_Results_Each_Epoch')):
    os.makedirs(os.path.join(output_dir, 'NeuroSim_Results_Each_Epoch'))

out = open(os.path.join(output_dir, "PythonWrapper_Output.csv"), 'ab')
out_firstline = np.array([["epoch", "average loss", "accuracy"]])
np.savetxt(out, out_firstline, delimiter=",", fmt='%s')
delta_distribution = open(os.path.join(output_dir, "delta_dist.csv"), 'ab')
delta_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean",
                           "7_mean", "8_mean", "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(delta_distribution, delta_firstline, delimiter=",", fmt='%s')
weight_distribution = open(os.path.join(output_dir, "weight_dist.csv"), 'ab')
weight_firstline = np.array([["1_mean", "2_mean", "3_mean", "4_mean", "5_mean", "6_mean",
                            "7_mean", "8_mean", "1_std", "2_std", "3_std", "4_std", "5_std", "6_std", "7_std", "8_std"]])
np.savetxt(weight_distribution, weight_firstline, delimiter=",", fmt='%s')
args.logdir = os.path.join(os.path.dirname(__file__), output_dir, args.logdir)
args = make_path.makepath(
    args, ['log_interval', 'test_interval', 'logdir', 'epochs'])
misc.logger.init(args.logdir, 'train_log_' + current_time)
logger = misc.logger.info
misc.ensure_dir(os.path.join(output_dir, args.logdir))
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))

logger("========================================")

if __name__ == "__main__":
    args.cuda = device != torch.device("cpu")
    logdir = os.path.join(os.path.dirname(__file__), output_dir, 'output')
    logger = misc.logger.info
    train_loader, _, test_loader = LoadCIFAR10(
        batch_size=batch_size, validation=False, num_workers=1)
    model = VGG8(args=args, logger=logger).to(device)
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
    best_acc, old_file = 0, None
    try:
        if args.cellBit != args.wl_weight:
            print("Warning: Weight precision should be the same as the cell precison !")

        paramALTP = {}
        paramALTD = {}
        k = 0
        for layer in list(model.parameters())[::-1]:
            d2dVariation = torch.normal(torch.zeros_like(
                layer), args.d2dVari * torch.ones_like(layer))
            NL_LTP = torch.ones_like(
                layer) * args.nonlinearityLTP + d2dVariation
            NL_LTD = torch.ones_like(
                layer) * args.nonlinearityLTD + d2dVariation
            paramALTP[k] = wage_quantizer.GetParamA(
                NL_LTP.cpu().numpy()) * args.max_level
            paramALTD[k] = wage_quantizer.GetParamA(
                NL_LTD.cpu().numpy()) * args.max_level
            k = k + 1

        df = pd.DataFrame(columns=['epoch', 'elapsed_time',
                                   'test_set_accuracy', 'test_loss'])
        run = neptune.init(
            project=neptune_project_name,
            api_token=neptune_api_token,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for epoch in range(args.epochs):
            velocity = {}
            i = 0
            for layer in list(model.parameters())[::-1]:
                velocity[i] = torch.zeros_like(layer)
                i = i + 1

            if epoch in decreasing_lr:
                grad_scale = grad_scale / 8.0

            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                indx_target = target.clone()
                if args.cuda:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = model(data)
                loss = wage_util.SSE(output, target)
                loss.backward()
                j = 0
                for name, param in list(model.named_parameters())[::-1]:
                    velocity[j] = gamma * velocity[j] + alpha * param.grad.data
                    param.grad.data = velocity[j]
                    if param.grad.data.abs().max() > 0:
                        param.grad.data = wage_quantizer.QG(param.data, args.wl_weight, param.grad.data, args.wl_grad, grad_scale,
                                                            torch.from_numpy(paramALTP[j]).cuda(), torch.from_numpy(paramALTD[j]).cuda(), args.max_level, args.max_level)
                    j = j + 1

                optimizer.step()
                for name, param in list(model.named_parameters())[::-1]:
                    param.data = wage_quantizer.W(
                        param.data, param.grad.data, args.wl_weight, args.c2cVari)

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]
                    correct = pred.cpu().eq(indx_target).sum()
                    acc = float(correct) * 1.0 / len(data)

            if epoch % args.test_interval == 0:
                test_loss = 0
                correct = 0
                model.eval()
                for i, (data, target) in enumerate(test_loader):
                    if i == 0:
                        hook_handle_list = hook.hardware_evaluation(
                            model, args.wl_weight, args.wl_activate, epoch)
                    indx_target = target.clone()
                    if args.cuda:
                        data = data.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                    with torch.no_grad():
                        output = model(data)
                        test_loss_i = wage_util.SSE(output, target)
                        test_loss += test_loss_i.data
                        pred = output.data.max(1)[1]
                        correct += pred.cpu().eq(indx_target).sum()
                    if i == 0:
                        hook.remove_hook_list(hook_handle_list)

                test_loss = test_loss / len(test_loader)
                acc = 100. * correct / len(test_loader.dataset)
                accuracy = acc.cpu().data.numpy()

            end.record()
            torch.cuda.synchronize()
            df = df.append({'epoch': epoch, 'elapsed_time': start.elapsed_time(
                end), 'test_set_accuracy': acc.item(), 'test_loss': test_loss.cpu().item()}, ignore_index=True)
            df.to_csv('Train_VGG8.csv', index=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        run.stop()
        print(df)
