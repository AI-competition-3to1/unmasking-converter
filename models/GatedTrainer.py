import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from utils.losses.losses import *
from utils.metrics.metrics import *
from utils.dataloder import *
from unmasking.gated_network import *
import time

class GatedTrainer():
    def __init__(self,epoch,data_path):
        
        self.num_epochs=epoch
        self.epochs=0
        self.iter=0
        self.print_per_iter=10
        self.visualize_per_iter= 500
        self.device = torch.device('cuda:0')
        self.train_path= os.path.join(data_path, 'train')
        self.trainset=dataset_for_gated(self.train_path)
        self.sample_folder='/content/drive/MyDrive/aicomp/sample'
        self.trainloader = data.DataLoader(
            self.trainset, 
            batch_size=2,
            collate_fn = self.trainset.collate_fn)

        
        self.epoch = 0
        self.iters = 0
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)
        self.device = torch.device('cuda:0')


        self.model_G = GatedGenerator().to(self.device)
        self.model_D = NLayerDiscriminator(3, use_sigmoid=False).to(self.device)
        self.model_P = PerceptualNet(name = "vgg16", resize=False).to(self.device)

        self.criterion_adv = GANLoss(target_real_label=0.9, target_fake_label=0.1)
        self.criterion_rec = nn.SmoothL1Loss()
        self.criterion_ssim = SSIM(window_size = 11)
        self.criterion_per = nn.SmoothL1Loss()

        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=1e-4)
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters(), lr=1e-4)

    def validate(self, sample_folder, sample_name, img_list):
        save_img_path = os.path.join(sample_folder, sample_name+'.png') 
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)

        # imgs shape: Bx5xCxWxH

        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow= 5)
        print(f"Save image to {save_img_path}")

    def fit(self):
        self.model_G.train()
        self.model_D.train()

        running_loss = {
            'D': 0,
            'G': 0,
            'P': 0,
            'R_1': 0,
            'R_2': 0,
            'T': 0,
        }

        running_time = 0
        step = 0
        for epoch in range(self.epoch, self.num_epochs):
          self.epoch = epoch
          for i, batch in enumerate(self.trainloader):
            start_time = time.time()
            imgs = batch['imgs'].to(self.device)
            masks = batch['masks'].to(self.device)

            # Train discriminator
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()
            
            first_out, second_out = self.model_G(imgs, masks)

            first_out_wholeimg = imgs * (1 - masks) + first_out * masks     
            second_out_wholeimg = imgs * (1 - masks) + second_out * masks

            masks = masks.cpu()

            fake_D = self.model_D(second_out_wholeimg.detach())
            real_D = self.model_D(imgs)

            loss_fake_D = self.criterion_adv(fake_D, target_is_real=False)
            loss_real_D = self.criterion_adv(real_D, target_is_real=True)

            loss_D = (loss_fake_D + loss_real_D) * 0.5

            loss_D.backward()
            self.optimizer_D.step()

            real_D = None

            # Train Generator
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()

            fake_D = self.model_D(second_out_wholeimg)
            loss_G = self.criterion_adv(fake_D, target_is_real=True)

            fake_D = None
            
            # Reconstruction loss
            loss_l1_1 = self.criterion_rec(first_out_wholeimg, imgs)
            loss_l1_2 = self.criterion_rec(second_out_wholeimg, imgs)
            loss_ssim_1 = self.criterion_ssim(first_out_wholeimg, imgs)
            loss_ssim_2 = self.criterion_ssim(second_out_wholeimg, imgs)

            loss_rec_1 = 0.5 * loss_l1_1 + 0.5 * (1 - loss_ssim_1)
            loss_rec_2 = 0.5 * loss_l1_2 + 0.5 * (1 - loss_ssim_2)

            # Perceptual loss
            loss_P  = self.model_P(second_out_wholeimg, imgs)                          
            lambda_G= 1.0
            lambda_rec_1= 100.0
            lambda_rec_2= 100.0
            lambda_per= 10.0
            loss = lambda_G * loss_G + lambda_rec_1 * loss_rec_1 + lambda_rec_2 * loss_rec_2 + lambda_per * loss_P
            loss.backward()
            self.optimizer_G.step()

            end_time = time.time()

            imgs = imgs.cpu()
            # Visualize number
            running_time += (end_time - start_time)
            running_loss['D'] += loss_D.item()
            running_loss['G'] += (lambda_G * loss_G.item())
            running_loss['P'] += (lambda_per * loss_P.item())
            running_loss['R_1'] += (lambda_rec_1 * loss_rec_1.item())
            running_loss['R_2'] += (lambda_rec_2 * loss_rec_2.item())
            running_loss['T'] += loss.item()
            

            if self.iters % self.print_per_iter == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                
                running_loss = {
                    'D': 0,
                    'G': 0,
                    'P': 0,
                    'R_1': 0,
                    'R_2': 0,
                    'T': 0,
                }
                running_time = 0
            # Visualize sample
            if self.iters % self.visualize_per_iter == 0:
                masked_imgs = imgs * (1 - masks) + masks
                
                img_list = [imgs, masked_imgs, first_out, second_out, second_out_wholeimg]
                #name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                filename = f"{self.epoch}_{str(self.iters)}"
                self.validate(self.sample_folder, filename , img_list)

            self.iters += 1
        torch.save(
                    self.model_G.state_dict(),
                    ('/content/drive/MyDrive/aicomp/gated.pt'))
        print('Model saved!')
            
                
                    