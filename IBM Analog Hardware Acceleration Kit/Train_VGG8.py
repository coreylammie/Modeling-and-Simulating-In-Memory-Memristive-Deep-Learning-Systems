import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
import ssl
import sys
import os
import random
import numpy as np
import pandas as pd
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.getcwd() + "/..")

import aihwkit
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.devices import ReRamSBPresetDevice

from VGG8 import VGG8
from LoadCIFAR import LoadCIFAR10


# -------------------------------------------------------
neptune_project_name = ''
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
batch_size = 128
epochs = 256
seed = 1
ADC_precision = 6
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


RPU_CONFIG = aihwkit.simulator.configs.configs.SingleRPUConfig(
    device=aihwkit.simulator.configs.devices.ReferenceUnitCell(
        unit_cell_devices=[
            ReRamSBPresetDevice(),
            ReRamSBPresetDevice(),
        ]
    ),
    backward=aihwkit.simulator.configs.utils.IOParameters(
        out_res=(1 / ADC_precision)),
    forward=aihwkit.simulator.configs.utils.IOParameters(
        out_res=(1 / ADC_precision)),
)

if __name__ == "__main__":
    train_loader, _, test_loader = LoadCIFAR10(
        batch_size=batch_size, validation=False, num_workers=1)
    model = VGG8(RPU_CONFIG).to(device)
    optimizer = AnalogSGD(model.parameters(),
                          lr=initial_learning_rate, momentum=momentum)
    optimizer.regroup_param_groups(model)
    criterion = torch.nn.CrossEntropyLoss()
    df = pd.DataFrame(columns=['epoch', 'elapsed_time',
                      'test_set_accuracy', 'test_loss'])
    run = neptune.init(
        project=neptune_project_name,
        api_token=neptune_api_token,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(epochs):
        if epoch in learning_rate_schedule:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10

        model.train()
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            try:
                loss.backward()
                optimizer.step()
            except:
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for _, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss
                pred = output.data.max(1)[1]
                correct += pred.eq(target.to(device).data.view_as(pred)
                                   ).cpu().sum()

        test_loss = test_loss / len(test_loader)
        acc = 100. * correct / len(test_loader.dataset)
        accuracy = acc.cpu().data.numpy()
        end.record()
        torch.cuda.synchronize()
        df = df.append({'epoch': epoch, 'elapsed_time': start.elapsed_time(
            end), 'test_set_accuracy': acc.item(), 'test_loss': test_loss.cpu().item()}, ignore_index=True)
        df.to_csv('Train_VGG8.csv', index=False)

    run.stop()
    print(df)
