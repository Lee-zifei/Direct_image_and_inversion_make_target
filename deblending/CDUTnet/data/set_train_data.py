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

import sys, os, platform
import os
import torch
import torch.distributed as dist
from myprog import myimtocol
import numpy as np
from myprog import *
import matplotlib.pyplot as plt

def generate_random_01_matrix(n1, n2,sparsity):
    """
    生成一个大小为 n1 x n2 的随机 0-1 矩阵
    """
    sampling_matrix = np.random.rand(n1, n2) * sparsity
    
    return sampling_matrix

def build_dataset():
    n1 = 2000
    n2 = 330
    n3 = 150
    
    clip = 0.001
    ii = 80
    # aelmp
    mm = seis(2)
    amplitude = 1000
    dither1 = amplitude * (2 * np.random.rand(n2) - 1)
    print(dither1)
    for j in range(n3):
        data1 = read_d2(f'../../../../re_forward/forward_output/{j+1}source_seismogram.bin',[n1,n2,1],0)
        data2 = read_d2(f'../../../../re_forward/forward_output/{j+150}source_seismogram.bin',[n1,n2,1],0)
        print(j)

        target_1 = data1
        target_2 = data2

        # print(time2.shape)
        

        
        inter2 = dither(target_2, dither1)
        inter1 = dither(target_1, -dither1)

        sample_1 = target_1 + inter2
        sample_2 = target_2 + inter1

        # np.save('./cosl_training/Train/target/a'+str(j)+'.npy',target_1)
        # np.save('./cosl_training/Train/target/b'+str(j)+'.npy',target_2)
        
        # np.save('./cosl_training/Train/sample_lev_1/a'+str(j)+'.npy',sample_1)
        # np.save('./cosl_training/Train/sample_lev_1/b'+str(j)+'.npy',sample_2)

        # np.save('./cosl_training/Train/sample_lev_2/a'+str(j)+'.npy',sample_1_1)
        # np.save('./cosl_training/Train/sample_lev_2/b'+str(j)+'.npy',sample_2_1)
        

            
        # fig = plt.figure(figsize=(16, 16),dpi=100)
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # ax3 = fig.add_subplot(221)
        # ax3.imshow(target_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)
        # ax4 = fig.add_subplot(222)
        # ax4.imshow(target_2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)
        # ax4 = fig.add_subplot(223)
        # ax4.imshow(sample_1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)
        # ax4 = fig.add_subplot(224)
        # ax4.imshow(sample_2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)       
        # plt.show()
        
        
        np.save('./Marmousi_retraining/Train/target/a'+str(j)+'.npy',target_1)
        np.save('./Marmousi_retraining/Train/target/b'+str(j)+'.npy',target_2)

        np.save('./Marmousi_retraining/Train/sample/a'+str(j)+'.npy',sample_1)
        np.save('./Marmousi_retraining/Train/sample/b'+str(j)+'.npy',sample_2)

def build_dataset_salt():
    n1 = 3000
    n2 = 330
    n3 = 150
    
    clip = 0.001
    ii = 80
    # aelmp
    mm = seis(2)
    amplitude = 1000
    dither1 = amplitude * (2 * np.random.rand(n2) - 1)
    print(dither1)
    for j in range(n3):
        data1 = read_d2(f'../../../../re_forward/salt3_forward_output/{j+1}source_seismogram.bin',[n1,n2,1],0)
        data2 = read_d2(f'../../../../re_forward/salt3_forward_output/{j+150}source_seismogram.bin',[n1,n2,1],0)
        print(j)

        target_1 = data1
        target_2 = data2

        inter2 = dither(target_2, dither1)
        inter1 = dither(target_1, -dither1)

        sample_1 = target_1 + inter2
        sample_2 = target_2 + inter1

        np.save('./salt_training/Train/target/a'+str(j)+'.npy',target_1)
        np.save('./salt_training/Train/target/b'+str(j)+'.npy',target_2)

        np.save('./salt_training/Train/sample/a'+str(j)+'.npy',sample_1)
        np.save('./salt_training/Train/sample/b'+str(j)+'.npy',sample_2)

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
        for i in range(1,150,50):
            print(i)
            targets = np.load('./dagang/Train/target/'+data_type+str(i)+'.npy')
            print(targets.shape)
            inputs = np.load('./dagang/Train/sample/'+data_type+str(i)+'.npy')
            # print (targets.shape)
            mm = seis(2)
            # print (np.max(targets))
            # fig = plt.figure(figsize=(16, 16),dpi=200)
            # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            # ax3 = fig.add_subplot(121)
            # ax3.imshow(targets, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)
        
            # ax4 = fig.add_subplot(122)
            # ax4.imshow(inputs, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)

            # plt.show()
if __name__ == '__main__':
#      bin2npy()
    # build_dataset_salt()
    fig()


