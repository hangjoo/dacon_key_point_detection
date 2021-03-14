import torch.nn as nn
from torchvision import models


def create_model(class_num: int, model_name: str):
    if model_name == "resnet":
        model_ft = models.resnet34(pretrained=True)

        model_in_features = model_ft.fc.in_features
        model_ft.fc = nn.linear(model_in_features, class_num)

        return model_ft
