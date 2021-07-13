import os
import torch
from tqdm import tqdm
from utils.config import get_config
from utils.dataset import MaskDataset, MaskDataLoader
from utils.images import plot_image, save_cropped_image
from utils.logger import Logger
from model import get_model, get_params


logger = Logger().get_logger()

MODEL_NAME = "Mask segmentating model"
MODEL_PATH = "mask_segment_model.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(config, data_loader):
    # Prepare pretrained model - faster rcnn
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH)
    else:
        model = get_model(config["model"])
        model.to(device)

    # Train Model
    EPOCHS, optimizer = get_params(config["model"], model)
    logger.info(f"Train `{MODEL_NAME}` \n (params : {config['model']})")

    batch_size = data_loader.batch_size
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(data_loader)
        for imgs, annotations in pbar:
            imgs = list(img.to(device) for img in imgs)
            annotations = list(
                {k: v.to(device) for k, v in t.items()} for t in annotations
            )
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            pbar.set_postfix({"Loss": f"{epoch_loss:.4f}"})

        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] loss {epoch_loss:.4f}")

        if config["save"]:
            loss_str = str(f"{epoch_loss:.4f}").replace(".", "_")
            basedir = config["model"]["directory"]
            filename = f"mask_detect_model-batch{batch_size}-ep{epoch}-{loss_str}.pt"
            filepath = os.path.join(basedir, filename)
            logger.info(f"Save the `{MODEL_NAME}` (path : {filepath})")
            torch.save(model, filename)

    if config["save"]:
        basedir = config["model"]["directory"]
        filepath = os.path.join(basedir, filename)
        logger.info(f"Save the `{MODEL_NAME}` (path : {filepath})")
        torch.save(model, filename)

    return model


def main(config):
    logger.info(f"Running on {device}")

    # Prepare Dataset
    logger.info(f"Load Dataset from {config['data']['directory']}")
    dataset = MaskDataset(config["data"])
    data_loader = MaskDataLoader(config["loader"], dataset).loader

    logger.info(f"Model : `{MODEL_NAME}` (mode: {config['mode']}) ")
    if config["mode"] == "train":
        model = train(config, data_loader)
    model = torch.load("pretrain/mask_detect_model-ep17-16_2006.pt")
    model.eval()

    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        preds = model(imgs)

        if config["mode"] == "train":
            annotations = [{k: v for k, v in t.items()} for t in annotations]

        # Show results
        for i in range(len(imgs)):
            if config["data"]["visualize"]:
                plot_image(imgs[i], preds[i], annotations[i])

            save_cropped_image(config["data"], imgs[i], preds[i])

    logger.info("Process Done")


if __name__ == "__main__":
    config = get_config()

    try:
        main(config)
    except KeyboardInterrupt:
        logger.warning(f"Abort! (KeyboardInterrupt)")
