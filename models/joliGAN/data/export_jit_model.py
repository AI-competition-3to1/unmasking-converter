import sys
import torch
import numpy as np
sys.path.append('../')
from models import networks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-in-file',help='file path to generator model to export (.pth file)',required=True)
parser.add_argument('--model-out-file',help='file path to exported model (.pt file)')
parser.add_argument('--model-type',default='resnet_attn',help='model type, e.g. mobile_resnet_9blocks')
parser.add_argument('--img-size',default=256,type=int,help='square image size')
parser.add_argument('--cpu',action='store_true',help='whether to export for CPU')
parser.add_argument('--bw',action='store_true',help='whether input/output is bw')
args = parser.parse_args()

if not args.model_out_file:
    model_out_file = args.model_in_file.replace('.pth','.pt')
else:
    model_out_file = args.model_out_file

if args.bw:
    input_nc = output_nc = 1
else:
    input_nc = output_nc = 3

ngf = 64
use_dropout = False
decoder = True
img_size = args.img_size
model = networks.define_G(input_nc,output_nc,ngf,args.model_type,'instance',use_dropout,
                          decoder=decoder,
                          img_size=args.img_size,
                          img_size_dec=args.img_size)
if not args.cpu:
    model = model.cuda()
    
model.eval()
model.load_state_dict(torch.load(args.model_in_file))

if args.cpu:
    device = 'cpu'
else:
    device = 'cuda'
dummy_input = torch.randn(1, input_nc, args.img_size, args.img_size, device=device)
jit_model = torch.jit.trace(model, dummy_input)
jit_model.save(model_out_file)
