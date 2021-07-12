from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Profile
from .config import _PATH_DIR
import os

# Create your views here.
def index(request):
    
    form=Profile()
    #form.title=request.POST['title']
    try:
        form.image=request.FILES['image']
    except: #이미지가 없어도 그냥 지나가도록-!
        pass
    form.save()
    profile=Profile.objects.all()
    profile=profile.last()

    path_dir= _PATH_DIR
    file_list = os.listdir(path_dir)
    print(_PATH_DIR)
    print(file_list)
    download_file = file_list[-1]
    
    download_path = 'images_converted/'+download_file
    print(type(download_path))
    return render(request,'convert/index.html', {'profile':profile, 'download_path':download_path})

#def upload(request):
#    return render(request,'convert/upload.html')

#def upload_create(request):
#    form=Profile()
#    #form.title=request.POST['title']
#    try:
#        form.image=request.FILES['image']
#    except: #이미지가 없어도 그냥 지나가도록-!
#        pass
#    form.save()
#    return redirect('/convert/profile/')   

#def profile(request):
#    profile=Profile.objects.all()
#    profile=profile.last()
#    return render(request,'convert/profile.html',{'profile':profile})