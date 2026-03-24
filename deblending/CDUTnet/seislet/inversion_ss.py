import collections
import numpy as np
import os
from myprog import myimtocol,dither
import torch
import torch.nn as nn
from torch._utils import _get_all_device_indices
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast,Scatter
from seislet import SyncMaster

_ChildMessage = collections.namedtuple('_ChildMessage', ['input', 'index'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['input'])

class Inversion(nn.Module):

	def __init__(self,config):
		super(Inversion, self).__init__()
		self._sync_master = SyncMaster(self._data_parallel_master)
		self._is_parallel = False
		self._parallel_id = None
		self._slave_pipe = None
		self.config = config
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
		output = inv_ss(inputs,self.config)
		B,C,H,W = output.shape
		output = output.cuda()
		num=B//len(target_gpus)
		broadcasted = Broadcast.apply(target_gpus,output)
		outputs = []
		for i, rec in enumerate(newmediates):
			outputs.append((rec[0], broadcasted[i][i*num:(i+1)*num,:,:,:]))  ###顺序有可能会错
		return outputs

abspath=os.getcwd()

def inv_ss(inputs,config): #### B C H W

	name=config.TEST.NAME
	H,W=config.DATA.IMG_SIZE
	sh,sw=config.TEST.XSLIDE,config.TEST.TSLIDE
	num1 = int(np.floor((H - 64) / sh) + 1 + (np.mod(H - 64, sh) != 0))
	num2 = int(np.floor((W - 64) / sw) + 1 + (np.mod(W - 64, sw) != 0))
	datasize = num1 * num2
	inputs = inputs[:datasize*2,:,:,:]
	b,c,h,w = inputs.shape
	d1de1 = myimtocol(inputs[:b//2,0,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,H,W,sh,sw,0)
	d2de1 = myimtocol(inputs[b//2:,0,:,:], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 0)
	obsers = np.load(abspath+'/'+config.TEST.TEST_PATH + '/sample_'+name+'.npy')
	delay = np.load(abspath+'/'+config.TEST.TEST_PATH + '/delay_'+name+'.npy')[:W]
	obser1, obser2 = obsers[0, :, :W], obsers[1, :, :W]
	temp1 = obser1 - dither(d2de1,delay)
	temp2 = obser2 - dither(d1de1,-delay)

	temp1 = myimtocol(temp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
	temp2 = myimtocol(temp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
	out = np.concatenate((temp1,temp2),axis=0)
	B,H,W = out.shape
	out = out.reshape(B,1,H,W)
	return torch.from_numpy(out)

