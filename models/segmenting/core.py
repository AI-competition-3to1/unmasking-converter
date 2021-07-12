import torch
from tqdm import tqdm
from utils.config import get_config
from utils.dataset import MaskDataset, MaskDataLoader
from utils.images import plot_image
from utils.logger import Logger
from model import get_model, get_params


logger = Logger().get_logger()

MODEL_NAME = "Mask segmentating model"
MODEL_PATH = "mask_segment_model.pt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(config, data_loader):
    # Prepare pretrained model - faster rcnn
    model = get_model(config["model"])
    model.to(device)

    # Train Model
    EPOCHS, optimizer = get_params(config["model"], model)
    logger.info(f"Train `{MODEL_NAME}` \n (params : {config['model']})")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for imgs, annotations in tqdm(data_loader):
            imgs = list(img.to(device) for img in imgs)
            annotations = list(
                {k: v.to(device) for k, v in t.items()} for t in annotations
            )
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses
            
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] loss {epoch_loss:.4f}")

        if config["save"]:
            loss_str = str(epoch_loss).replace(".", "-")
            filename = f"mask_segment_model-{loss_str}.pt"
            logger.info(f"Save the `{MODEL_NAME}` (path : {filename})")
            torch.save(model, filename)

    if config["save"]:
        logger.info(f"Save the `{MODEL_NAME}` (path : {MODEL_PATH})")
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
    elif config["mode"] == "pretrain":
        model = torch.load(MODEL_PATH)
    model.eval()

    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        preds = model(imgs)

        annotations = [{k: v for k, v in t.items()} for t in annotations]
        # Show results
        for i in range(len(imgs)):
            plot_image(imgs[i], preds[i], annotations[i])

        torch.cuda.empty_cache()

    logger.info("Process Done")


if __name__ == "__main__":
    config = get_config()

    try:
        main(config)
    except KeyboardInterrupt:
        logger.warning(f"Abort! (KeyboardInterrupt)")
