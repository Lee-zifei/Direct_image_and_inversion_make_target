#================================================================
#    Copyright (C) 2019 Chengdu University of Technology.
#    Copyright (C) 2019 Shaohuan Zu.
#
#    Filename:     ztest.py
#    Author：      Shaohuan Zu
#    Institute：   Chengdu University of Technology
#    Email：       zushaohuan19@cdut.edu.cn
#    Date：        2019/11/25    15:45:29
#    Function：
#
#    This program is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or any later version.
#================================================================
import numpy as np
import os
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
# from subfunctions import *
from subfunctions import normalize_patches,denormalize_patches,seis,npy2bin,mutter

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from myprog import *
# import keras.backend.tensorflow_backend as KTF #
import warnings
import argparse
import os
from pynvml import *
import tensorflow as tf
from keras import backend as K


########################obtain parameters#################################
parser = argparse.ArgumentParser(description='read parameters')
parser.add_argument ('--model', default=None, type=int, help='Choose model')
parser.add_argument ('--abs', default=10, type=int,help='Noise abs')

args = parser.parse_args()
kk = args.model
# Abs = args.abs

########################set gpu information#################################

nvmlInit()
GpuCount = nvmlDeviceGetCount()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# KTF.set_session(session )

########################set data parameters ###############################
batch_size = 16       #batch size for training in each step
n1=2000;                 # the training volumn size   nz
n2=330;                # the training volumn sie    nt
patch_rows = 128        # patch size
patch_cols= 128         # patch size
tslide=1               # time strid
xslide=1              # space stride




filename1 = './data_test/hyper.dat'
hyper = np.fromfile(filename1, dtype='float32')
hyper = hyper.reshape(n2,n1).T

filename2 = './data_test/hyper2.dat'
hyper2 = np.fromfile(filename2, dtype='float32')
hyper2 = hyper2.reshape(n2,n1).T


filename3 = './data_test/obser.dat'
X_test1 = np.fromfile(filename3, dtype='float32')
X_test1 = X_test1.reshape(n2,n1).T

filename4 = './data_test/obser2.dat'
X_test2 = np.fromfile(filename4, dtype='float32')
X_test2 = X_test2.reshape(n2,n1).T

filename5 = './data_test/dither.dat'
delay = np.fromfile(filename5, dtype='float32')



########################load the trained model#################################



########################Iterative deblending#################################

abs_list = ['05','10']
for Abs in abs_list:
# for Abs in range(5,6,1):
    # Abs=10
    def snr(y_true, y_pred):
        return y_pred
    model = load_model(f'../training_marmousi/model_{Abs}_%d.h5'%kk,custom_objects={'snr':snr})
    # model = load_model(f'../training1/model_01_%d.h5'%kk,custom_objects={'snr':snr})
    print ('##################################################################################\n')
    print ('#########################Loading model is {}######################################\n'.format(Abs))
    print ('##################################################################################\n')
    temp1 = X_test1
    temp2 = X_test2
    niter = 10
    clip = 0.01
    mm = seis(2)
    snr1 = np.zeros((niter,1), dtype='float32')
    snr2 = np.zeros((niter,1), dtype='float32')
    for i in range(niter):
        reobser1 = myimtocol(temp1, patch_rows, patch_cols, n1, n2, tslide, tslide, 1)
        reobser2 = myimtocol(temp2, patch_rows, patch_cols, n1, n2, tslide,tslide, 1)
        X1_test = np.concatenate((reobser1,reobser2),axis=0)
        [test_datasize, patch_rows, patch_cols] = X1_test.shape

        X1_test = X1_test.reshape(test_datasize, patch_rows, patch_cols,1)
        X1_test, X1_test_stats = normalize_patches(X1_test)
        
        Y1_test = model.predict(X1_test, batch_size=batch_size, verbose=1)
        Y1_test = np.squeeze(Y1_test)
        [nn1, nn2, nn3]  = Y1_test.shape
        Y1_test = denormalize_patches(Y1_test, X1_test_stats)
        
        d1de1 = myimtocol(Y1_test[0:int(nn1/2),:,:], patch_rows, patch_cols, n1, n2, tslide,tslide, 0)
        d2de1 = myimtocol(Y1_test[int(nn1/2):nn1,:,:], patch_rows, patch_cols, n1, n2, tslide,tslide, 0)
        
        d1de1 = mutter(d1de1,301,-30,8)
        d2de1 = mutter(d2de1,29,-30,8)
        
        temp1 = X_test1 - dither (d2de1, delay);
        temp2 = X_test2 - dither (d1de1, -delay);

        temp1 = mutter(temp1,301,-30,8)
        temp2 = mutter(temp2,29,-30,8)
        snr1[i] = calculate_snr(hyper, temp1)
        snr2[i] = calculate_snr(hyper2, temp2)

        print ('iteration =%d,snr = %.2f'%((i+1),snr1[i]))
        # npy2bin(f"./data_test/output/temp1_{i}.dat",d1de1)
        # npy2bin(f"./data_test/output/temp2_{i}.dat",d2de1)

        plt.figure(dpi=200)
        plt.subplot(141);plt.imshow(X_test1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
        plt.title('iteration =%d'%(i+1))
        plt.subplot(142);plt.imshow(X_test2,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
        plt.title('iteration =%d'%(i+1))
        plt.subplot(143);plt.imshow(temp1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
        plt.title('iteration =%d'%(i+1))
        plt.subplot(144);plt.imshow(temp2,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
        plt.title('iteration =%d'%(i+1))
        plt.show()
    # npy2bin(f"./data_test/output/snrs_{Abs}.dat",snr1)
    # npy2bin(f"./data_test/output/temp1_{i}_abs_{Abs}.dat",temp1)

    # plt.figure(dpi=200)
    # plt.subplot(141);plt.imshow(X_test1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
    # plt.title('iteration =%d'%(i+1))
    # plt.subplot(142);plt.imshow(hyper,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
    # plt.title('iteration =%d'%(i+1))
    # plt.subplot(143);plt.imshow(temp1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
    # plt.title('iteration =%d'%(i+1))
    # plt.subplot(144);plt.imshow(hyper - temp1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.2)
    # plt.title('iteration =%d'%(i+1))
    # plt.show()
    # npy2bin(f"./data_test/output/snr2.dat",snr2)










