import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassCalibrationError
import pandas as pd
import json


def get_param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    return total_params


def evaluate(model, dataloader, criterion, num_classes, device):
    model.eval()
    results = {}
    model_loss = 0.0
    correct = 0
    total = 0
    ece = 0
    ece = MulticlassCalibrationError(num_classes, n_bins=15, norm="l1")
    for data, target in dataloader:
        with torch.no_grad():
            if data.device != device:
                data = data.to(device)
            if target.device != device:
                target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            ece.update(torch.softmax(output, dim=1), target)
            model_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    results["model_loss"] = model_loss / len(dataloader)
    results["ece"] = ece.compute().item()
    results["accuracy"] = 100 * (correct / total)
    return results


def get_device():
    if torch.cuda.is_availabel():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"


def get_function(model, data_loader, num_classes, criterion, device):
    model.eval()
    model_loss = 0.0
    correct = 0
    total = 0
    predictions = pd.DataFrame()
    temp = pd.DataFrame()
    ece = MulticlassCalibrationError(num_classes, n_bins=15, norm="l1")
    results = {}
    for index, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            ece.update(output, target)
            softmax = torch.softmax(output, dim=1)
            temp = pd.DataFrame(softmax.cpu().numpy())
            predictions = pd.concat([predictions, temp], ignore_index=True)
            model_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    results["accuracy"] = 100 * (correct / total)
    results["model_loss"] = model_loss / len(data_loader)
    results["ece"] = ece.compute().item()
    results["function"] = predictions.to_numpy().tolist()
    return results


def load_json(path):
    with open(path) as f:
        dict = json.load(f)
    return dict


