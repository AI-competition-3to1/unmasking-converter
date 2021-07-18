from django.shortcuts import render, redirect
from django.http import HttpResponse
from numpy.core.fromnumeric import resize
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
from PIL import Image
from torchvision import transforms

from models.unet import UNet
from models.gated_network import GatedGenerator
from logger import Logger
from convert.bbox import scale_bbox, box_scaler

logger = Logger().get_logger()

# Create your views here.
def index(request):

    ind = True

    form = Profile()
    # form.title=request.POST['title']
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
        {"profile": profile, "download_path": download_path, "index": ind},
    )


# def convert(request):
#     ind = False

#     profile = Profile.objects.all()
#     profile = profile.last()

#     path_dir = _PATH_DIR

#     model_in_file = (
#         BASE_DIR
#         + "\\models\\joliGAN\\checkpoints\\face_masks_removal\\latest_net_G_A.pt"
#     )

#     img_in = BASE_DIR + "\\app\\" + profile.image.url
#     logger.info(BASE_DIR + "\\app\\" + profile.image.url)
#     logger.info(img_in)

#     img_out = _PATH_DIR + "\\" + profile.image.url.split("/")[-1]
#     img_size = 256

#     model = torch.jit.load(model_in_file)

#     #########################################
#     # if you use gpu
#     model = model.cuda()

#     # reading image
#     img = cv2.imread(img_in)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (img_size, img_size))

#     # preprocessing
#     tranlist = [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
#     tran = transforms.Compose(tranlist)
#     img_tensor = tran(img)
#     #########################################
#     # if you use gpu
#     img_tensor = img_tensor.cuda()

#     # print('tensor shape=',img_tensor.shape)

#     # run through model
#     out_tensor = model(img_tensor.unsqueeze(0))[0].detach()
#     # print(out_tensor)
#     # print(out_tensor.shape)

#     # post-processing
#     out_img = out_tensor.data.cpu().float().numpy()
#     out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
#     out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(img_out, out_img)
#     print("Successfully generated image ", img_out)

#     # ind = True
#     download_path = "images_converted/" + profile.image.url.split("/")[-1]

#     return render(
#         request,
#         "convert/index.html",
#         {"profile": profile, "download_path": download_path, "index": ind},
#     )


def detect(profile):
    detect_path = BASE_DIR + "\\app\\weights\\detect.pt"
    model = torch.load(detect_path)
    model.eval()

    img_path = f"{BASE_DIR}\\app\\{profile.image.url}"
    image = Image.open(img_path).convert("RGB")
    
    transform = transforms.ToTensor()
    img = transform(image).unsqueeze(0)
    img = img.to(torch.device("cuda"))

    pred = model(img)[0]
    pred = scale_bbox(pred)
    img = np.array(img[0].cpu().data.permute(1, 2, 0))
    pred = pred["boxes"].cpu().data
    
    boxes = list()
    imgs = list()
    for box in pred:
        xmin, ymin, xmax, ymax = box_scaler(box)["box"]

        bbox = (xmin, ymin, min(img.shape[1], xmax), min(img.shape[0], ymax))
        xmin, ymin, xmax, ymax = bbox

        cropped_img = img[ymin:ymax, xmin:xmax]
        if cropped_img.shape[0] < 25:
            logger.info(cropped_img.shape)
            continue

        convert_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
        resized_img = convert_img.resize((256, 256), Image.LANCZOS)
        
        boxes.append(bbox)
        imgs.append(resized_img)
        
    return boxes, imgs


def unmask(request):
    ind = False
    profile = Profile.objects.all()
    profile = profile.last()
    
    img_path = f"{BASE_DIR}\\app\\{profile.image.url}"
    origin_image = Image.open(img_path).convert("RGB")
    
    unet_path = BASE_DIR + "\\app\\weights\\unet.pt"
    gated_conv_path = BASE_DIR + "\\app\\weights\\gated.pt"
    myUnet = UNet()
    myUnet.load_state_dict(torch.load(unet_path))
    
    count = 0
    boxes, imgs = detect(profile)
    for box, img in zip(boxes, imgs):
        filedir = f"{BASE_DIR}/app/source/"
        filename = f"test{count}.png"
        filepath = os.path.join(filedir, filename)
        count += 1
        
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # save_path = _PATH_DIR + "\\" + profile.image.url.split("/")[-1]
        image = torch.FloatTensor(np.expand_dims(image, axis=0) / 255.0).permute(0, 3, 1, 2)

        seg = myUnet(image)
        seg_n = seg.detach().permute(0, 2, 3, 1).detach().numpy()
        seg_n = np.around(np.reshape(seg_n, (256, 256, 1))) * 255
        seg_n = cv2.erode(seg_n, np.ones((3, 3)), 1)
        seg_n = cv2.dilate(seg_n, np.ones((5, 5)), 1)

        seg = np.expand_dims(seg_n, axis=0)
        seg = np.expand_dims(seg, axis=0)
        seg = torch.FloatTensor(seg / 255)

        myGnet = GatedGenerator()
        myGnet.load_state_dict(torch.load(gated_conv_path))
        l = myGnet(image, seg)
        g = make_grid(image * (1 - seg) + l[1] * seg).permute(1, 2, 0).detach().numpy()
        g *= 255
        g = np.reshape(g, (256, 256, 3))

        img = cv2.cvtColor(np.uint8(g), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, 'RGB')
        
        height = box[3] - box[1]
        width = box[2] - box[0]

        img = img.resize((width, height))
        img.save(filepath)
        logger.info(f"Pearson : {count}")

        origin_image.paste(img, (box[0], box[1]))
   
    # ########
    save_path = _PATH_DIR+ '\\' + profile.image.url.split('/')[-1]
    origin_image.save(save_path)

    download_path = "images_converted/" + profile.image.url.split("/")[-1]
    
    return render(
        request,
        "convert/index.html",
        {"profile": profile, "download_path": download_path, "index": ind},
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
