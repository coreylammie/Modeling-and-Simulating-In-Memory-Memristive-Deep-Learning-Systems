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
from aihwkit.simulator.presets.devices import ReRamSBPresetDevice

from LoadCIFAR import LoadCIFAR10
from GoogLeNet import GoogLeNet


# -------------------------------------------------------
neptune_project_name = ''
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
batch_size = 128
seed = 1
ADC_precision = 6
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
    _, _, test_loader = LoadCIFAR10(
        batch_size=batch_size, validation=False, num_workers=1
    )
    model = GoogLeNet(RPU_CONFIG).to(device)
    model.eval()
    df = pd.DataFrame(columns=['idx', 'elapsed_time',
                      'test_set_accuracy', 'test_loss'])
    run = neptune.init(
        project=neptune_project_name,
        api_token=neptune_api_token,
    )
    with torch.no_grad():
        for i in range(10):
            test_loss = 0
            correct = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                test_loss += F.cross_entropy(output, target).data
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

            test_loss = test_loss / len(test_loader)
            acc = 100. * correct / len(test_loader.dataset)
            accuracy = acc.cpu().data.numpy()
            end.record()
            torch.cuda.synchronize()
            df = df.append({'idx': i, 'elapsed_time': start.elapsed_time(
                end), 'test_set_accuracy': acc.item(), 'test_loss': test_loss.cpu().item()}, ignore_index=True)
            df.to_csv('Inference_GoogLeNet.csv', index=False)

    run.stop()
    print(df)
