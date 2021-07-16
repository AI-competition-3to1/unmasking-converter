from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Profile
from configs.config import _PATH_DIR, BASE_DIR
import os
import sys
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np


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
