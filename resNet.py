import torch.nn as nn
import torch
import torchvision.models as models


def create_resNet(classes):
    model = models.resnet34(pretrained=True)

    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, classes),
        nn.Softmax(dim=1)
    )

    return model