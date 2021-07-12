import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd 
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utils.config import get_config
from utils.dataset import MaskDataset, MaskDataLoader
from utils.images import plot_image
from utils.logger import Logger


logger = Logger().get_logger()

MODEL_NAME = 'Mask segmentating model'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(config):
    # Run Process
    logger.info(f"Run `{MODEL_NAME} model` (mode: {config['mode']}) on {device}")

    # Prepare Dataset
    logger.info(f"Load Dataset from {config['data']['directory']}")
    dataset = MaskDataset(config["data"])
    data_loader = MaskDataLoader(config["loader"], dataset).loader

    # Prepare pretrained model - faster rcnn
    NUM_CLASSES = 3
    logger.info(f"Set Faster R-CNN model (num_classes : {NUM_CLASSES})")

    model = get_model_instance_segmentation(num_classes=NUM_CLASSES)
    model.to(device)


    logger.info(f"Load Dataset (BATCH_SIZE : {data_loader.batch_size})")
    # for imgs, annotations in data_loader:
    #     imgs = list(img.to(device) for img in imgs)
    #     annotations = list({k: v.to(device) for k, v in t.items()} for t in annotations)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = list({k: v.to(device) for k, v in t.items()} for t in annotations)
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            epoch_loss += losses
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] loss {epoch_loss:.4f}")
     
    logger.info(model.eval())
    model.eval()
    preds = model(imgs)

    print("Prediction")
    plot_image(imgs[0], preds[0])

    print("Target")
    plot_image(imgs[0], annotations[0])

    print("Done")

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    config = get_config()

    try:
        main(config)
    except KeyboardInterrupt:
        logger.warning(f"Abort! (KeyboardInterrupt)")
