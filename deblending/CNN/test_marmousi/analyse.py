# ==================================================================================
#    Copyright (C) 2025 Chengdu University of Technology.
#    Copyright (C) 2025 Zifei LI.
#    
#    Filename：analyse.py
#    Author：Zifei LI
#    Institute：Chengdu University of Technology
#    Email：2024010196@stu.cdut.edu.cn
#    Work：2025/05/08/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python/subfuctions' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
    myprog_path_survey='/home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import * 
import numpy as np
import matplotlib.pyplot as plt
clip = 0.02
mm = seis(2)

n1=3000;                 # the training volumn size   nz
n2=330;                # the training volumn sie    nt

data1 = read_d2('./data_test/output/temp1_9_abs_5.dat',[n1,n2,1],0)
data2 = read_d2('./data_test/output/temp1_9_abs_10.dat',[n1,n2,1],0)
data_y = read_d2('./data_test/hyper.dat',[n1,n2,1],0)
data_b = read_d2('./data_test/obser.dat',[n1,n2,1],0)
plt.figure(dpi=200)
plt.subplot(221);plt.imshow(data1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
plt.subplot(222);plt.imshow(data2,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
plt.subplot(223);plt.imshow(data_y-data1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
plt.subplot(224);plt.imshow(data_y-data2,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)

plt.show()