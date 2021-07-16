import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import vgg19
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from UNetTrainer import *
from models.unet  import *


EPOCHS = 25
trainer=UnetTrainer(EPOCHS,'D:/downloads/aicomp/unmasking-converter/models/data/')

trainer.fit()

myunet=UNet()
myunet.load_state_dict(torch.load('D:/downloads/aicomp/unmasking-converter/models/data/unet.pt'))
myunet.eval()