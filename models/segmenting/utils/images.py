import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(img_tensor, annotation, block=True):
    fig, ax = plt.subplots(1)
    
    img = img_tensor.cpu().data
    annotation["boxes"] = annotation["boxes"].cpu().data 
    annotation["labels"] = annotation["labels"].cpu().data 

    # Display the image
    ax.imshow( np.array( img.permute(1, 2, 0) ) )

    for box, label in zip( annotation["boxes"], annotation["labels"] ):
        # print("label", label)
        # print("box", box)
        xmin, ymin, xmax, ymax = box
        # Create a Rectangle patch
        edgecolor = 'b' if label == 1 else 'r'
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor=edgecolor,facecolor='none')

        # Add the patch to the Axes 
        ax.add_patch(rect)
        ax.axis("off")
    plt.show(block=block)