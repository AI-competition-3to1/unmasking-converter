import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(img, preds, annos, block=True):
    fig = plt.figure(figsize=(16, 8))
    rows = 1
    cols = 2

    img = img.cpu().data
    preds["boxes"] = preds["boxes"].cpu().data
    preds["labels"] = preds["labels"].cpu().data

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(np.array(img.permute(1, 2, 0)))
    ax1.set_title("Prediction")
    ax1.axis("off")
    draw_patch(ax1, preds)

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(np.array(img.permute(1, 2, 0)))
    ax2.set_title("Target")
    ax2.axis("off")
    draw_patch(ax2, annos)

    plt.show(block=block)
    plt.close()


def draw_patch(ax, annotation, block=True):
    for box, label in zip(annotation["boxes"], annotation["labels"]):
        # print("label", label)
        # print("box", box)
        xmin, ymin, xmax, ymax = box
        # Create a Rectangle patch
        edgecolor = "b" if label == 1 else "r"
        rect = patches.Rectangle(
            (xmin, ymin),
            (xmax - xmin),
            (ymax - ymin),
            linewidth=1,
            edgecolor=edgecolor,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
    return ax
