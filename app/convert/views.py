from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Profile
from configs.config import _PATH_DIR, BASE_DIR
import os
import sys
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
from models.gated_network import GatedGenerator

# Create your views here.
def index(request):

    ind = True

    form = Profile()
    #form.title=request.POST['title']
    try:
        form.image = request.FILES["image"]
    except:  # 이미지가 없어도 그냥 지나가도록-!
        pass
    form.save()
    profile = Profile.objects.all()
    profile = profile.last()

    path_dir = _PATH_DIR
    file_list = os.listdir(path_dir)
    
    try:
        download_file = file_list[0]
    except:  # 이미지가 없어도 그냥 지나가도록-!
        download_file = ""
    download_path = "images_converted/" + download_file

    return render(
        request,
        "convert/index.html",
        {"profile": profile, "download_path": download_path, "index" : ind},
    )

def convert(request):

    ind = False

    profile = Profile.objects.all()
    profile = profile.last()

    path_dir = _PATH_DIR
    file_list = os.listdir(path_dir)

    model_in_file = BASE_DIR+'\\models\\joliGAN\\checkpoints\\face_masks_removal\\latest_net_G_A.pt'
    img_in = BASE_DIR + '\\app\\' + profile.image.url
    print(img_in)

    
    img_out = _PATH_DIR+ '\\' + profile.image.url.split('/')[-1]
    print(img_out.split('\\')[-1])
    img_size = 256

    model = torch.jit.load(model_in_file)

    #########################################
    # if you use gpu
    model = model.cuda()

    # reading image
    img = cv2.imread(img_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size,img_size))

    # preprocessing
    tranlist = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    tran = transforms.Compose(tranlist)
    img_tensor = tran(img)
    #########################################
    # if you use gpu
    img_tensor = img_tensor.cuda()
    
    #print('tensor shape=',img_tensor.shape)

    # run through model
    out_tensor = model(img_tensor.unsqueeze(0))[0].detach()
    #print(out_tensor)
    #print(out_tensor.shape)

    # post-processing
    out_img = out_tensor.data.cpu().float().numpy()
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_out,out_img)
    print('Successfully generated image ',img_out)

    #ind = True

    download_path = "images_converted/" + profile.image.url.split('/')[-1]

    return render(
        request,
        "convert/index.html",
        {"profile": profile, "download_path": download_path, "index" : ind},
    )

def unmask(request):

    profile = Profile.objects.all()
    profile = profile.last()

    unet_path = BASE_DIR + "\\models\\unmasking\\weights\\unet.pt"
    gated_conv_path = BASE_DIR + "\\weights\\gated.pt"
    img = BASE_DIR + '\\app\\' + profile.image.url
    save_path = _PATH_DIR+ '\\' + profile.image.url.split('/')[-1]
    
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

    return render(
        request,
        "convert/index.html",
        {"profile": profile, "download_path": download_path, "index" : ind},
    )

# def upload(request):
#    return render(request,'convert/upload.html')

# def upload_create(request):
#    form=Profile()
#    #form.title=request.POST['title']
#    try:
#        form.image=request.FILES['image']
#    except: #이미지가 없어도 그냥 지나가도록-!
#        pass
#    form.save()
#    return redirect('/convert/profile/')

# def profile(request):
#    profile=Profile.objects.all()
#    profile=profile.last()
#    return render(request,'convert/profile.html',{'profile':profile})
