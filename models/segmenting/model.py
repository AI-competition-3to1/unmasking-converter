import os
import torch
from torch.optim import SGD
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(config):
    num_classes = config["num_classes"]

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_params(config, model):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = Adam(
        params,
        lr=config["lr"],
    )

    epochs = config["epochs"]

    return epochs, optimizer
