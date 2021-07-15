# Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

# Getting Started

## Installation
  - For pip users, please type the command `pip install -r requirements.txt`.
  - If you use gpu while training, 
    - CUDA 10.2, `pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
    - CUDA 11.1, `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
  - If you use cpu while training, `pip install torch torchvision torchaudio`

## JoliGAN train

### Position of the dataset
  - In the ./data/trainA Folder : Img_masked dataset
  - In the ./data/trainB Folder : Img_unmasked dataset

### train code
  - `visdom`
  - another cmd `python train.py --dataroot ./data/ --model cycle_gan --pool_size 50 --no_dropout --no_rotate --name face_masks_removal --n_epochs 20 --n_epochs_decay 20`

### Store the model trained
  - You can find `./checkpoints/face_masks_removal/latest_net_G_A.pth` model
  - Convert .pth file to .pt file
  - If you use gpu, `python export_jit_model.py --model-in-file ./checkpoints/face_masks_removal/latest_net_G_A.pth --img-size 256 --n_epochs 100 --n_epochs_decay 100`
  - If you use cpu, `python export_jit_model.py --model-in-file ./checkpoints/face_masks_removal/latest_net_G_A.pth --img-size 256 --cpu`

### Test
  - If you use cpu, `python gen_jit_single_image.py --model-in-file ./checkpoints/face_masks_removal/latest_net_G_A.pt --img-size 256 --img-in /path/to/img_domain_A.png --img-out /path/to/img_domain_B.png --cpu`
  - If you use gpu, `python gen_jit_single_image.py --model-in-file ./checkpoints/face_masks_removal/latest_net_G_A.pt --img-size 256 --img-in /path/to/img_domain_A.png --img-out /path/to/img_domain_B.png`
