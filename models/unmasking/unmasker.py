import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.unet import UNet
from models.gated_network import GatedGenerator

def unmask(unet_path,gated_conv_path,img,save_path):
    myUnet=UNet()
    myUnet.load_state_dict(torch.load(unet_path))
    image=cv2.imread(img, cv2.IMREAD_COLOR)
    image=cv2.resize(image,(256,256))
    image=torch.FloatTensor(np.expand_dims(image,axis=0)/255.0).permute(0,3,1,2)
    seg=myUnet(image)
    seg_n=seg.detach().permute(0,2,3,1).detach().numpy()
    seg_n=np.around(np.reshape(seg_n,(256,256,1)))*255
    seg_n=cv2.erode(seg_n,np.ones((3,3)),1)
    seg_n=cv2.dilate(seg_n,np.ones((5,5)),1)
    seg=np.expand_dims(seg_n,axis=0)
    seg=np.expand_dims(seg,axis=0)
    seg=torch.FloatTensor(seg/255)
    myGnet= GatedGenerator()
    myGnet.load_state_dict(torch.load(gated_conv_path))
    l=myGnet(image,seg)
    g=make_grid(image*(1-seg)+l[1]*seg).permute(1,2,0).detach().numpy()
    g*=255
    cv2.imwrite(save_path,g)