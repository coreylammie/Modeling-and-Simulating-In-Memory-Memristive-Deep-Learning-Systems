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

import memtorch
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities

from LoadCIFAR import LoadCIFAR10
from GoogLeNet import GoogLeNet


# -------------------------------------------------------
neptune_project_name = ''
neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
batch_size = 128
seed = 1
ADC_precision = 6
weight_precision = 6
R_ON = 1e4
R_ON_std = 1e3
R_OFF = 1e5
R_OFF_std = 1e4
tile_shape = (16, 16)
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    _, _, test_loader = LoadCIFAR10(
        batch_size=batch_size, validation=False, num_workers=1,
    )
    model = GoogLeNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'r_off': memtorch.bh.StochasticParameter(loc=R_OFF, scale=R_OFF_std, min=1),
                                  'r_on': memtorch.bh.StochasticParameter(loc=R_ON, scale=R_ON_std, min=1)}
    patched_model = patch_model(copy.deepcopy(model),
                                memristor_model=reference_memristor,
                                memristor_model_params=reference_memristor_params,
                                module_parameters_to_patch=[
                                    torch.nn.Linear, torch.nn.Conv2d],
                                mapping_routine=naive_map,
                                transistor=True,
                                programming_routine=None,
                                tile_shape=tile_shape,
                                scaling_routine=naive_scale,
                                ADC_resolution=ADC_precision,
                                ADC_overflow_rate=0.,
                                quant_method='linear')
    patched_model_ = apply_nonidealities(copy.deepcopy(patched_model),
                                         non_idealities=[
                                             memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates],
                                         conductance_states=weight_precision)
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
                output = patched_model_(data)
                test_loss += criterion(output, target)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

            test_loss = test_loss / len(test_loader)
            acc = 100. * correct / len(test_loader.dataset)
            accuracy = acc.cpu().data.numpy()
            end.record()
            torch.cuda.synchronize()
            df = df.append({'idx': i, 'elapsed_time': start.elapsed_time(
                end), 'test_set_accuracy': acc.item(), 'test_loss': test_loss.cpu().item()}, ignore_index=True)
            df.to_csv('Inference GoogLeNet.csv', index=False)

    run.stop()
    print(df)
