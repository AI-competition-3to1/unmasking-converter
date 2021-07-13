import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from utils.bbox import scale_bbox, box_scaler


def plot_image(img, preds, annos, block=True):
    fig = plt.figure(figsize=(24, 8))
    rows = 1
    cols = 3

    img = np.array(img.cpu().data.permute(1, 2, 0))
    preds["boxes"] = preds["boxes"].cpu().data
    preds["labels"] = preds["labels"].cpu().data

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img)
    ax1.set_title("Target")
    ax1.axis("off")
    draw_patch(ax1, preds)

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(img)
    ax2.set_title("Prediction")
    ax2.axis("off")
    draw_patch(ax2, preds)

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(img)
    ax3.set_title("Filtering")
    ax3.axis("off")
    draw_patch(ax3, scale_bbox(preds), pred=True)

    plt.show(block=block)
    plt.close()


def save_cropped_image(config, img, preds):
    basedir = config["output_directory"]

    for box, label in zip(preds["boxes"], preds["labels"]):
        xmin, ymin, xmax, ymax = box_scaler(box)["box"]
        cropped_img = img[
            max(0, ymin) : min(img.shape[0], ymax),
            max(0, xmin) : min(img.shape[1], xmax),
        ]

        # Ignore Too small images
        if cropped_img.shape[0] < 20:
            continue

        convert_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
        resized_img = convert_img.resize((256, 256), Image.LANCZOS)

        if label == 1:
            file_count = len(os.listdir(basedir))
            filename = f"facemask_{file_count}.jpg"
            filepath = os.path.join(basedir, filename)
            resized_img.save(filepath)


def draw_patch(ax, annotation, pred=False):
    for box, label in zip(annotation["boxes"], annotation["labels"]):
        xmin, ymin, xmax, ymax = box_scaler(box)["box"] if pred else box
        linewidth = 1 if xmax - xmin < 20 and ymax - ymin < 20 else 3
        # Create a Rectangle patch
        edgecolor = "b" if label == 1 else "r"
        rect = patches.Rectangle(
            (xmin, ymin),
            (xmax - xmin),
            (ymax - ymin),
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
    return ax
