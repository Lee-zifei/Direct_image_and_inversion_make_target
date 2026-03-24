# -*- coding: utf-8 -*-
# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.
import time
import collections
import contextlib
import numpy as np
import os
from myprog import myimtocol
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from data.Data.input.seislet_transform import  invbst
from torch._utils import _get_all_device_indices
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast,Scatter

from seislet import SyncMaster



_ChildMessage = collections.namedtuple('_ChildMessage', ['input', 'index'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['input'])


class inverse_seislet(nn.Module):

	def __init__(self,config):
		super(inverse_seislet, self).__init__()

		self._sync_master = SyncMaster(self._data_parallel_master)
		self._is_parallel = False
		self._parallel_id = None
		self._slave_pipe = None
		self.config=config

	def forward(self, input,index):
	# Reduce-and-broadcast the statistics.
		if self._parallel_id == 0:
			output = self._sync_master.run_master(_ChildMessage(input,index))
		else:
			output = self._slave_pipe.run_slave(_ChildMessage(input,index))
		return output

	def __data_parallel_replicate__(self, ctx, copy_id): ###not need to modify
		self._is_parallel = True
		self._parallel_id = copy_id

		# parallel_id == 0 means master device.
		if self._parallel_id == 0:
			ctx.sync_master = self._sync_master   ####device 0 为主人
		else:
			self._slave_pipe = ctx.sync_master.register_slave(copy_id) ###其他的GPU为奴隶

	def _data_parallel_master(self, intermediates):###这里写输入之间要干啥
		"""Reduce the sum and square-sum, compute the statistics, and broadcast it."""
		index=intermediates[0][1][1].detach().cpu().numpy()
		newmediates=[[] for i in range(len(intermediates))]
		for inp in intermediates:
			i=inp[1][0][0,0,0,0].get_device()
			newmediates[i]=[inp[0],inp[1][0]] ###input drop the index ###sorted by device ID

		for i, inp in enumerate(newmediates):
			if i==0:
				inputs=inp[1].detach().cpu().numpy()
			else:
				temp=inp[1].detach().cpu().numpy()
				inputs=np.concatenate((inputs,temp),axis=0)
		target_gpus = [i for i in range(len(newmediates))]
		output= invbst(inputs, index,self.config)
		B,C,H,W=output.shape
		output=output.cuda()
		num=B//len(target_gpus)
		broadcasted = Broadcast.apply(target_gpus,output)
		outputs = []
		for i, rec in enumerate(newmediates):
			outputs.append((rec[0], broadcasted[i][i*num:(i+1)*num,:,:,:]))  ###顺序有可能会错
		return outputs











abspath=os.getcwd()

def newinvbst(a,index,config): #### B C H W
	name=index[0,0]
	name=str(int(name))
	# vmax=index[0,1]
	a=a[:,0,:,:]
	hyper=myimtocol(a,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,0)
	# hyper=hyper*vmax
	hyper=hyper.T
	hyper=np.ascontiguousarray(hyper,dtype='float32')
	f = open(abspath+'/data/Data/swap/'+name+'_pseislet.dat','wb') ### 网络处理后输出成dat
	f.write(hyper)
	f.close()
	invprocess2dat(name,config.DATA.ROW,config.DATA.COL,test=config.TESTF) ### 从seislet域转为时间域
	datas = dat2numpy(abspath+'/data/Data/swap/'+name + "_denoise", size=(config.DATA.ROW,config.DATA.COL)) ## 从dat转成numpy
	return datas

def invbst(a,index,config): #### B C H W
	name=index[0,0]
	name=str(int(name))
	# vmax=index[0,1]
	a=a[:,0,:,:]
	hyper=myimtocol(a,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,0)
	# hyper=hyper*vmax
	hyper=hyper.T
	hyper=np.ascontiguousarray(hyper,dtype='float32')
	f = open(abspath+'/data/Data/swap/'+name+'_pseislet.dat','wb') ### 网络处理后输出成dat
	f.write(hyper)
	f.close()
	invprocess2dat(name,config.DATA.ROW,config.DATA.COL,test=config.TESTF) ### 从seislet域转为时间域
	datas = dat2numpy(abspath+'/data/Data/swap/'+name + "_denoise", size=(config.DATA.ROW,config.DATA.COL)) ## 从dat转成numpy
	out = myimtocol(datas,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
	B,H,W = out.shape
	out = out.reshape(B,1,H,W)
	datast = torch.from_numpy(out)
	return datast

def dat2numpy(name,size=(1024,512)):
	img=np.fromfile(name+'.dat',dtype='float32')
	img=img.reshape(size[1],size[0]).T
	return img

def invprocess2dat(name,n1,n2,d1=0.004,d2=1,o1=0,o2=0,test=False):
	os.system('''echo n1=%i n2=%i d1=%g d2=%g o1=%g o2=%g data_format="native_float" esize=4 in=%s_pseislet.dat > %s_pseislet.rsf'''%(n1, n2, d1, d2, o1, o2,
				abspath+'/data/Data/swap/'+name, abspath+'/data/Data/swap/'+name))
	if test:
		os.system(
			'''sfseislet < %s_pseislet.rsf > %s_denoise.rsf dip=%s_dip.rsf type=b order=1 eps=1 inv=y unit=y ''' % (
			abspath + '/data/Data/swap/' + name, abspath + '/data/Data/swap/' + name,
			abspath + '/data/Data/swap/' + name))
	else:
		os.system('''sfseislet < %s_pseislet.rsf > %s_denoise.rsf dip=%s_dip.rsf type=b order=1 eps=1 inv=y unit=y'''%(
			abspath+'/data/Data/swap/'+name, abspath+'/data/Data/swap/'+name,
			abspath+'/data/Data/input/dip/'+name))
	os.system('''< %s_denoise.rsf sfdd type=float | sfrsf2bin > %s_denoise.dat'''%(abspath+'/data/Data/swap/'+name,abspath+'/data/Data/swap/'+name))

