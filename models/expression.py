import torch
import torch.nn as nn
import torchvision.models as models


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def build_expression_model():
    """ResNet-34 — SAMA PERSIS dengan training FER2013"""
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5 * 0.6),
        nn.Linear(256, 7)
    )
    return model


def load_expression_model(weight_path, device):
    model = build_expression_model()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model