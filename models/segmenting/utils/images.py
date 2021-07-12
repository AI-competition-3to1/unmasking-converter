import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(img_tensor, annotation, block=True):
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data
    # Display the image
    ax.imshow( np.array( img.permute(1, 2, 0) ) )

    for box, label in zip( annotation["boxes"], annotation["labels"] ):
        print("label",label)
        print("box",box)
        xmin, ymin, xmax, ymax = box
        # Create a Rectangle patch
        if label==1:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')
        else:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes 
        ax.add_patch(rect)
        ax.axis("off")
    plt.show(block=block)