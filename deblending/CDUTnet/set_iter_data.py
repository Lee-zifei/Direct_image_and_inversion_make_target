# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：stetrain.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/06/07/
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
from subfunctions import dither,snr,seis,read_d3,bin2npy_3d,mutter,read_d2,normalize_patches,denormalize_patches,myimtocol_0 as myimtocol

def generate_random_01_matrix(n1, n2,sparsity):
    """
    生成一个大小为 n1 x n2 的随机 0-1 矩阵
    """
    sampling_matrix = np.random.rand(n1, n2) * sparsity
    
    return sampling_matrix

import sys, os, platform
import os
import torch
import torch.distributed as dist
from myprog import myimtocol
import numpy as np
from myprog import *
import matplotlib.pyplot as plt
from logger import create_logger
from models import build_model
import argparse
from config import get_config
from collections import OrderedDict
from seislet import patch_replication_callback, DataParallelWithCallback


import time
def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/WUDTnet.yaml', metavar="FILE", help='path to config file' )
	parser.add_argument('--batch-size', type=int, default=1, help="batch size for single GPU")
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
	# blend_csg2_col = myimtocol(blend_csg2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
 
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
		# data_loader2=build_test_loader(config,blend_csg2_col)
		# for idx, (datas,_)  in enumerate(data_loader2):
		# 	datas=datas.cuda()

		# 	output2 = model(datas)####datasize,C H W
		# 	output2=output2.cpu().numpy()
		# 	if idx==0:
		# 		results2=output2
		# 	else:
		# 		results2=np.concatenate((results2,output2),axis=0)

	outputs1 = np.squeeze(results1)
 
	# ll = 1
	# outputs1[:, :ll, :] = 0        # 前10行
	# outputs1[:, -ll:, :] = 0       # 后10行
	# outputs1[:, :, :ll] = 0        # 前10列
	# outputs1[:, :, -ll:] = 0       # 后10列
 
	# outputs2 = np.squeeze(results2)
	# print(outputs1.shape)
 
	outputs1_denomo = denormalize_patches(outputs1, blend_csg1_col_stats)
	d1de1 = myimtocol(outputs1_denomo, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
	# d2de1 = myimtocol(outputs2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)

	# return d1de1,d2de1
	return d1de1

def build_dataset():
    _, config = parse_option()
    logger = create_logger(output_dir='./',name=f"{config.MODEL.TYPE}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    n1 = 3585
    n2 = 564
    n3_1 = 801
    n3_2 = 802
    
    clip = 0.005
    ii = 80
    # aelmp
    mm = seis(2)
    for j in range(n2):
        # print(j)
        blend_1 = np.load('./data/cosl_training/Train/sample_lev_2/a'+str(j)+'.npy')
        blend_2 = np.load('./data/cosl_training/Train/sample_lev_2/b'+str(j)+'.npy')
        
        target_1 = np.load('./data/cosl_training/Train/target/a'+str(j)+'.npy')
        target_2 = np.load('./data/cosl_training/Train/target/b'+str(j)+'.npy')
        
        time1 = bin2npy_3d('./blended_data/zsh_delay_1to2.dat',1,1,n3_1)
        time1 = time1.reshape(-1)   
        time1 = np.delete(time1, np.s_[0:36], axis=0)  
        time2 = bin2npy_3d('./blended_data/zsh_delay_2to1.dat',1,1,n3_1)
        time2 = time2.reshape(-1)
        time2 = np.delete(time2, np.s_[0:36], axis=0)  
        d1_blend = blend_1
        d2_blend = blend_2
        
        d1t = np.zeros_like(d2_blend)
        d2t = np.zeros_like(d2_blend)
        
        d1_new = np.zeros_like(d2_blend)
        d2_new = np.zeros_like(d2_blend)
        niter = 1
        for iter in range(niter):
            domain = 'CRG'

            # inter2 = dither(d2_new, time2)
            # inter1 = dither(d1_new, time1)

            d1t = d1_blend - dither(d2_new, time2-2000)-dither(d2_new, time2-3000)
            d2t = d2_blend - dither(d1_new, time1-2000)-dither(d1_new, time1-3000)

            SROW = 64
            SCOL = 64
            d1de1 = iter_one_times(config,domain,d1t,SROW,SCOL)
            d2de2 = iter_one_times(config,domain,d2t,SROW,SCOL)

            d1_new = d1de1
            d2_new = d2de2
            # snr1 =  snr(d1de1*max1, d1)
            # snr2 =  snr(d2de2*max1, d2)
            logger.info(f"正在设置{domain}第{str(j+1)}/{str(n2)}道训练集, 第{str(iter+1)}次迭代")
        
        
        d1tt = d1_blend - dither(d2_new, time2-2000)-dither(d2_new, time2-3000)
        d2tt = d2_blend - dither(d1_new, time1-2000)-dither(d1_new, time1-3000)
        # fig = plt.figure(figsize=(16, 16),dpi=100)
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # ax3 = fig.add_subplot(231)
        # ax3.imshow(d1tt, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(232)
        # ax4.imshow(target_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(233)
        # ax4.imshow(target_1-d1tt, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(234)
        # ax4.imshow(d2tt, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(235)
        # ax4.imshow(target_2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)    
        # ax4 = fig.add_subplot(236)
        # ax4.imshow(target_2-d2tt, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)      
        # plt.show()


        np.save('./data/cosl_training/Train/sample_lev_2_iter/a'+str(j)+'.npy',d1tt)
        np.save('./data/cosl_training/Train/sample_lev_2_iter/b'+str(j)+'.npy',d2tt)
            
        # fig = plt.figure(figsize=(16, 16),dpi=100)
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # ax3 = fig.add_subplot(221)
        # ax3.imshow(target_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(222)
        # ax4.imshow(target_2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(223)
        # ax4.imshow(sample_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)
        # ax4 = fig.add_subplot(224)
        # ax4.imshow(sample_1_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.3)       
        # plt.show()

def fig():
    clip = 0.005
    ii = 80
    # aelmp
    for j in range(1,3,1):
        if j == 1:
            data_type = 'a'
            ed_nm = 400
        elif j == 2:
            data_type = 'b'
            ed_nm = 140
        for i in range(1,500,50):
            # print(i)
            targets = np.load('./data/cosl_training/Train/target/'+data_type+str(i)+'.npy')
            # print(targets.shape)
            inputs = np.load('./data/cosl_training/Train/sample_lev_2_iter/'+data_type+str(i)+'.npy')
            # print (targets.shape)
            mm = seis(2)
            # print (np.max(targets))
            fig = plt.figure(figsize=(16, 16),dpi=200)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            ax3 = fig.add_subplot(121)
            ax3.imshow(targets, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            ax4 = fig.add_subplot(122)
            ax4.imshow(inputs, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)

            plt.show()
if __name__ == '__main__':
#      bin2npy()
    #build_dataset()
    fig()


