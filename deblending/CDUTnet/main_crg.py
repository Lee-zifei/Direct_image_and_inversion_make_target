import numpy as np
import os
# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：main_crg.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/08/21/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
    myprog_path_survey='/home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import *
from subfunctions import dither,snr,seis,read_d3,bin2npy_3d,mutter,read_d2,npy2bin, normalize_patches,denormalize_patches,myimtocol as myimtocol

import torch
from logger import create_logger
from models import build_model
# import os
# from metrics import AverageMeter,snr
import argparse
import numpy as np
from config import get_config
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from models import build_model
import os
import argparse
import numpy as np
from config import get_config
from collections import OrderedDict
from seislet import patch_replication_callback, DataParallelWithCallback
from models.NAFNet_arch import  NAFNet

import time
def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/WUDTnet.yaml', metavar="FILE", help='path to config file' )
	parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
	parser.add_argument('--data-path', type=str, help='path to dataset')

	parser.add_argument('--resume', help='resume from checkpoint')
	parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
	parser.add_argument('--use-checkpoint', action='store_true',
			    help="whether to use gradient checkpointing to save memory")
	parser.add_argument('--output', default='output', type=str, metavar='PATH',
			    help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
	parser.add_argument('--tag', help='tag of experiment')
	parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
	parser.add_argument('--throughput', action='store_true', help='Test throughput only')
	parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
	parser.add_argument(
		"--opts",
		help="Modify config options by adding 'KEY VALUE' pairs. ",
		default=None,
		nargs='+',
	)
	args, unparsed = parser.parse_known_args()
	config = get_config(args)
	return args, config

def input_data(filed,seeds):
	hyper = filed
	n1,n2=hyper.shape
	hyper1 = np.zeros((n1,n2))
	# hyper2 = np.zeros((n1,n2))
	hyper1 = hyper
	# hyper2 = hyper[1,:,:]
	
	# hyper = turnone(hyper)
	# np.random.seed(seeds)
	# hyper2 = np.flip(hyper, axis=1)
	# delay=np.random.randint(-100,100,n2,dtype='int16')
 
	# return hyper1,hyper2
	return hyper1

def build_test_loader(config,inputs):

    inputs = np.reshape(inputs, (1,)+inputs.shape)
    inputs = inputs.transpose(1,0,2,3)
    inputs = torch.from_numpy(inputs)

    dataset_test= torch.utils.data.TensorDataset(inputs,inputs)
    data_loader_test= torch.utils.data.DataLoader(
        dataset_test,
        batch_size=256,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return data_loader_test

def iter_one_times(config,style,filed,SROW,SCOL):

 
	seed = 10
	blend_csg1 = input_data(filed,seed)
	ROW,COL = blend_csg1.shape
	## data input 3000*256->3000*96
	n1, n2 = blend_csg1.shape

	model = build_model(config)
	torch.backends.cudnn.benchmark = False

	if style=='CSG':
		checkpoint_dict = torch.load('output/'+config.MODEL.TYPE+'_'+style+'_randonsample/default/ckpt_epoch_199.pth', map_location='cpu')['model']
	elif style == 'CRG':
		checkpoint_dict = torch.load('output/'+config.MODEL.NAME+'/default/ckpt_epoch_199.pth', map_location='cpu')['model']
	model.load_state_dict(checkpoint_dict, strict=True)
	model.cuda()
	model = DataParallelWithCallback(model)
	model.eval()

	blend_csg1_col = myimtocol(blend_csg1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
	blend_csg1_col_nomo,blend_csg1_col_stats = normalize_patches(blend_csg1_col)

 
	with torch.no_grad():
		data_loader1 = build_test_loader(config,blend_csg1_col_nomo)
		for idx, (datas,_) in enumerate(data_loader1):
			datas = datas.cuda()
			output1 = model(datas)####datasize,C H W
			output1 = output1.cpu().numpy()
			if idx==0:
				results1=output1
			else:
				results1=np.concatenate((results1,output1),axis=0)
	outputs1 = np.squeeze(results1)
	print(outputs1.shape)
 
	outputs1_denomo = denormalize_patches(outputs1, blend_csg1_col_stats)
	d1de1 = myimtocol(outputs1_denomo, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)

	return d1de1

def iter_one_times_iter_1(config,style,filed,SROW,SCOL):

 
	seed = 10
	blend_csg1 = input_data(filed,seed)
	ROW,COL = blend_csg1.shape
	## data input 3000*256->3000*96
	n1, n2 = blend_csg1.shape

	model = build_model(config)
	torch.backends.cudnn.benchmark = False

	if style=='CSG':
		checkpoint_dict = torch.load('output/'+config.MODEL.TYPE+'_'+style+'_randonsample/default/ckpt_epoch_199.pth', map_location='cpu')['model']
	elif style == 'CRG':
		checkpoint_dict = torch.load('output/'+config.MODEL.NAME+'_iter/default/ckpt_epoch_199.pth', map_location='cpu')['model']
	model.load_state_dict(checkpoint_dict, strict=True)
	model.cuda()
	model = DataParallelWithCallback(model)
	model.eval()

	blend_csg1_col = myimtocol(blend_csg1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
	blend_csg1_col_nomo,blend_csg1_col_stats = normalize_patches(blend_csg1_col)
 
	with torch.no_grad():
		data_loader1 = build_test_loader(config,blend_csg1_col_nomo)
		for idx, (datas,_) in enumerate(data_loader1):
			datas = datas.cuda()
			output1 = model(datas)####datasize,C H W
			output1 = output1.cpu().numpy()
			if idx==0:
				results1=output1
			else:
				results1=np.concatenate((results1,output1),axis=0)

	outputs1 = np.squeeze(results1)
 
	print(outputs1.shape)
 
	outputs1_denomo = denormalize_patches(outputs1, blend_csg1_col_stats)
	d1de1 = myimtocol(outputs1_denomo, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)

	return d1de1


if __name__ == '__main__':
	_, config = parse_option()
	datapath = './'
	logger = create_logger(output_dir=datapath,name=f"{config.MODEL.TYPE}")
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config.GPU)
	os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6'

	n1=2000;                 # the training volumn size   nz
	n2=330;                # the training volumn sie    nt
 
	filename1 = '../test_data1/data_test/hyper.dat'
	hyper = np.fromfile(filename1, dtype='float32')
	hyper = hyper.reshape(n2,n1).T

	filename2 = '../test_data1/data_test/hyper2.dat'
	hyper2 = np.fromfile(filename2, dtype='float32')
	hyper2 = hyper2.reshape(n2,n1).T


	filename3 = '../test_data1/data_test/obser.dat'
	X_test1 = np.fromfile(filename3, dtype='float32')
	X_test1 = X_test1.reshape(n2,n1).T

	filename4 = '../test_data1/data_test/obser2.dat'
	X_test2 = np.fromfile(filename4, dtype='float32')
	X_test2 = X_test2.reshape(n2,n1).T

	filename5 = '../test_data1/data_test/dither.dat'
	delay = np.fromfile(filename5, dtype='float32')

	clip = 0.01
	times = 20
	mm = seis(2)



	niter = 10

	for i in range(1):
		trace = i
		d1_new = np.zeros((n1,n2))
		d2_new = np.zeros((n1,n2))
		
		d1t = np.zeros((n1,n2))
		d2t = np.zeros((n1,n2))

		temp_input = np.zeros((2,n1,n2))
		d1_blend = X_test1
		d2_blend = X_test2

		d1t = X_test1
		d2t = X_test2
		for iter in range(niter):
			domain = 'CRG'
			# delay = delay_2d[trace,:]



			SROW = 16
			SCOL = 16
			d1de1 = iter_one_times(config,domain,d1t,SROW,SCOL)
			d2de2 = iter_one_times(config,domain,d2t,SROW,SCOL)

			n = 0.5
			A = 1.5

   
			logger.info(f"正在进行第{str(iter+1)}次迭代，当前处理到{domain}第{str(i+1)}/{str(n2)}道")


			temp1 = dither(d2de2, delay)
			temp2 = dither(d1de1, -delay)




			d1_new= d1de1
			d2_new= d2de2

			inter1 = dither(d2_new, delay)
			inter2 = dither(d1_new, -delay)

			d1t = d1_blend         - inter1
			d2t = d2_blend         - inter2
   
			# fig = plt.figure(dpi=200)
			# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
			# asp = 0.4
			# ax1 = fig.add_subplot(131)
			# ax1.imshow(d1_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

			# ax1 = fig.add_subplot(132)
			# ax1.imshow(d1t,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

			# ax1 = fig.add_subplot(133)
			# ax1.imshow(d1_new,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

			# # ax1 = fig.add_subplot(144)
			# # ax1.imshow(hyper1-d1_new,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

			# plt.show()
			# d1t= mutter(d1t,301,-30,8)
			# d2t= mutter(d2t,29,-30,8)
		npy2bin(f"./deblending_stack/d1t_{config.MODEL.NAME}_{trace}_layer1_{iter}.dat",d1t)
		npy2bin(f"./deblending_stack/d2t_{config.MODEL.NAME}_{trace}_layer1_{iter}.dat",d2t)
		npy2bin(f"./deblending_stack/d1new_{config.MODEL.NAME}_{trace}_layer1_{iter}.dat",d1_new)
		npy2bin(f"./deblending_stack/d2new_{config.MODEL.NAME}_{trace}_layer1_{iter}.dat",d2_new)
			# d2_new[:, :-1] = (d2_t+d2_new[:, :-1])/2
			# d1_new[:, :-1] = (d1_t+d1_new[:, :-1])/2
			# npy2bin('d1_t'+str(i)+'.dat',d1_t)
			# # npy2bin('d2_t.dat',d2_t)
			# npy2bin('d1_new'+str(i)+'.dat',d1_new)
			# # npy2bin('d2_new.dat',d2_new)
		# fig = plt.figure(dpi=200)
		# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
		# asp = 0.4
		# ax1 = fig.add_subplot(141)
		# ax1.imshow(d2_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax1 = fig.add_subplot(142)
		# ax1.imshow(d2t,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax1 = fig.add_subplot(143)
		# ax1.imshow(d2_new,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax1 = fig.add_subplot(144)
		# ax1.imshow(hyper2-d2_new,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# plt.show()


