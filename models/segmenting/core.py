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
from utils.dataset import MaskDataset
from utils.images import plot_image
from utils.logger import Logger

MODEL_NAME = 'Mask segmentating model'
logger = Logger().get_logger()


def main(config):
    logger.info(f"Run `{MODEL_NAME} model` (mode: {config['mode']})")

    imgs = list(sorted(os.listdir("../data/mask/images/")))
    labels = list(sorted(os.listdir("../data/mask/annotations/")))
    data_transform = transforms.Compose([transforms.ToTensor(), ])
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    dataset = MaskDataset(data_transform, datadir="../data/mask/")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(3)

    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        break

    num_epochs = 25
    model.to(device)
        
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
    #         print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
            epoch_loss += losses
        print(epoch_loss)
        
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        break 
       
    model.eval()
    preds = model(imgs)
    preds

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
