import torch
import torch.nn as nn
import os
from utils.losses.losses import *
from utils.metrics.metrics import *
from utils.dataloder import *
from unmasking.unet import *
import time
class UnetTrainer():
  def __init__(self,epoch,data_path):
    self.num_epochs=epoch
    self.epochs=0
    self.iter=0
    self.print_per_iter=10
    self.device = torch.device('cuda:0')
    self.train_path= os.path.join(data_path, 'train')
    self.validate_path= os.path.join(data_path,'validate')
    self.trainset=dataset_for_unet(self.train_path)
    self.validset=dataset_for_unet(self.validate_path)
    self.trainloader = data.DataLoader(
            self.trainset, 
            batch_size=10,
            collate_fn = self.trainset.collate_fn)

    self.valloader = data.DataLoader(
            self.validset, 
            batch_size=10,
            collate_fn = self.validset.collate_fn)
    self.criterion_dice = DiceLoss()
    self.criterion_bce = nn.BCELoss()
    self.epoch = 0
    self.iters = 0
    self.num_iters = (self.num_epochs+1) * len(self.trainloader)
    self.model=UNet().to(self.device)
    self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.0001)
  def train_epoch(self):
        self.model.train()
        running_loss = {
                'DICE': 0,
                'BCE':0,
                 'T': 0,
            }
        running_time = 0

        for idx, batch in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            inputs = batch['imgs'].to(self.device)
            targets = batch['masks'].to(self.device)
            
            start_time = time.time()
            
            outputs = self.model(inputs)

            loss_bce = self.criterion_bce(outputs, targets)
            loss_dice = self.criterion_dice(outputs, targets)
            loss = loss_bce + loss_dice
            loss.backward()
            self.optimizer.step()
            
            end_time = time.time()
            
            running_loss['T'] += loss.item()
            running_loss['DICE'] += loss_dice.item()
            running_loss['BCE'] += loss_bce.item()
            running_time += end_time-start_time

            self.iters +=1
  def validate_epoch(self):
        #Validate
        
        self.model.eval()
        metrics = [ PixelAccuracy(1),Iou(1)]
        running_loss = {
            'DICE': 0,
            'BCE':0,
             'T': 0,
        }

        running_time = 0
        print('=============================EVALUATION===================================')
        with torch.no_grad():
            start_time = time.time()
            for idx, batch in enumerate(self.valloader):
                
                inputs = batch['imgs'].to(self.device)
                targets = batch['masks'].to(self.device)
                outputs = self.model(inputs)
                loss_bce = self.criterion_bce(outputs, targets)
                loss_dice = self.criterion_dice(outputs, targets)
                loss = loss_bce + loss_dice
                running_loss['T'] += loss.item()
                running_loss['DICE'] += loss_dice.item()
                running_loss['BCE'] += loss_bce.item()
                for metric in metrics:
                    metric.update(np.around(outputs.cpu()), np.around(targets.cpu()))
            end_time = time.time()
            running_time += (end_time - start_time)
            running_time = np.round(running_time, 5)
            for key in running_loss.keys():
                running_loss[key] /= len(self.valloader)
                running_loss[key] = np.round(running_loss[key], 5)

            loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
            
            print('[{}/{}] || Validation || {} || Time: {}s'.format(self.epoch, self.num_epochs, loss_string, running_time))
            for metric in metrics:
                print(metric)
            print('==========================================================================')
  def fit(self):
    for epoch in range(self.epoch, self.num_epochs+1): 
      self.epoch = epoch
      self.train_epoch()
      self.validate_epoch()
    torch.save(
                    self.model.state_dict(),
                    ('/content/drive/MyDrive/aicomp/unet.py'))
    print('Model saved!')