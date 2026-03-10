import torch
import torch.nn as nn
import torchvision.models as models


FATIGUE_CLASSES = ['Fatigue', 'NonFatigue']


def build_fatigue_model():
    """ResNet-18 — SAMA PERSIS dengan training Fatigue"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.45),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.45 * 0.7),
        nn.Linear(128, 2)
    )
    return model


def load_fatigue_model(weight_path, device):
    model = build_fatigue_model()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model