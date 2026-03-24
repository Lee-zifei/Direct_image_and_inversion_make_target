import collections
import numpy as np
import os
from myprog import myimtocol,dither
import torch
import torch.nn as nn
from torch._utils import _get_all_device_indices
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast,Scatter
import copy
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
		output = inv(inputs, index,self.config)
		B,C,H,W = output.shape
		output = output.cuda()
		num=B//len(target_gpus)
		broadcasted = Broadcast.apply(target_gpus,output)
		outputs = []
		for i, rec in enumerate(newmediates):
			outputs.append((rec[0], broadcasted[i][i*num:(i+1)*num,:,:,:]))  ###顺序有可能会错
		return outputs

abspath=os.getcwd()
def inv(inputs,index,config): #### B C H W
	if len(index)==4:
		first=True
	else:
		first=False
	name=str(int(index[1]))
	num=int(index[0])
	flag1=False
	flag2=False
	test=False
	if num==0 or num == 5:
		prefix='p'
		H,W,sh,sw=1024, 354, 46, 58
		test = False if num == 0 else True
	elif num == 1 or num == 6:
		prefix = 'm'
		H,W,sh,sw=725,207,32,32
		test = False if num == 1 else True
	elif num == 2 or num == 7:
		prefix = 'l'
		H,W,sh,sw=900,300,40,48
		test = False if num == 2 else True
	elif num == 3 or num == 8:
		prefix = 'a'
		H,W,sh,sw=1200,120,56,48
		flag1 = True
		test = False if num == 3 else True
	elif num == 4 or num == 9:
		prefix = 'e'
		H,W,sh,sw=1200,151,27,48
		test = False if num == 4 else True
	# elif num == 10:
	# 	H, W, sh, sw = 1024, 512, 32, 35
	num1 = int(np.floor((H - 64) / sh) + 1 + (np.mod(H - 64, sh) != 0))
	num2 = int(np.floor((W - 64) / sw) + 1 + (np.mod(W - 64, sw) != 0))
	datasize = num1 * num2
	inputs = inputs[:datasize*2,:,:,:]
	b,c,h,w = inputs.shape
	d1de1 = myimtocol(inputs[:b//2,0,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,H,W,sh,sw,0)
	d2de1 = myimtocol(inputs[b//2:,0,:,:], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 0)
	if test:
		obsers = np.load(abspath+'/'+config.DATA.TEST_PATH + '/sample/'+prefix+name+'.npy')
		delay = np.load(abspath+'/'+config.DATA.TEST_PATH + '/delay/'+prefix+name+'.npy')[:W]
	# elif config.TEST.MODE:
	# 	obsers = np.load(abspath+'/test/'+config.TEST.TYPE + '/data/' + config.TEST.TYPE  + 'obs.npy')
	# 	delay = np.load(abspath+'/test/'+config.TEST.TYPE + '/data/delay.npy')[:W]
	else:
		obsers = np.load(abspath+'/'+config.DATA.DATA_PATH + '/sample/' + prefix+name + '.npy')
		delay = np.load(abspath+'/'+config.DATA.DATA_PATH + '/delay/' + prefix+name + '.npy')[:W]
	obser1, obser2 = obsers[0, :, :W], obsers[1, :, :W]
	temp1 = obser1 - dither(d2de1,delay)
	temp2 = obser2 - dither(d1de1,-delay)
	if not first:
		if test:
			c1 = np.load(abspath+'/' + config.DATA.TEST_PATH + '/swap2/'+prefix+name+'_1.npy',temp1)
			c2 = np.load(abspath + '/' + config.DATA.TEST_PATH + '/swap2/' + prefix + name + '_2.npy',temp2)
		# elif config.TEST.MODE:
		# 	c1 = np.load(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_1.npy')
		# 	c2 = np.load(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_2.npy')
		else:
			c1 = np.load(abspath+'/' + config.DATA.DATA_PATH + '/swap2/'+prefix+name+'_1.npy',temp1)
			c2 = np.load(abspath + '/' + config.DATA.DATA_PATH + '/swap2/' + prefix + name + '_2.npy',temp2)
		loss1 = L1loss(obser1, c1 + dither(c2, delay))
		loss1 += L1loss(obser2, c2 + dither(c1, -delay))
		loss1 /= 2
		loss = L1loss(obser1, temp1 + dither(temp2, delay))
		loss += L1loss(obser2, temp2 + dither(temp1, -delay))
		loss /=2
		if loss >  loss1:
			temp1,temp2 = c1,c2
		else:
			if test:
				np.save(abspath + '/' + config.DATA.TEST_PATH + '/swap2/' + prefix + name + '_1.npy',temp1)
				np.save(abspath + '/' + config.DATA.TEST_PATH + '/swap2/' + prefix + name + '_2.npy',temp2)
			# elif config.TEST.MODE:
			# 	np.save(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_1.npy')
			# 	np.save(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_2.npy')
			else:
				np.save(abspath + '/' + config.DATA.DATA_PATH + '/swap2/' + prefix + name + '_1.npy',temp1)
				np.save(abspath + '/' + config.DATA.DATA_PATH + '/swap2/' + prefix + name + '_2.npy',temp2)

	else:
		if test:
			np.save(abspath + '/' + config.DATA.TEST_PATH + '/swap/' + prefix + name + '_1.npy', temp1)
			np.save(abspath + '/' + config.DATA.TEST_PATH + '/swap/' + prefix + name + '_2.npy', temp2)
		# elif config.TEST.MODE:
		# 	np.save(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_1.npy',temp1)
		# 	np.save(abspath + '/test/' + config.TEST.TYPE + '/data/' + config.TEST.TYPE + '_2.npy',temp2)
		else:
			np.save(abspath + '/' + config.DATA.DATA_PATH + '/swap/' + prefix + name + '_1.npy', temp1)
			np.save(abspath + '/' + config.DATA.DATA_PATH + '/swap/' + prefix + name + '_2.npy', temp2)
		loss = L1loss(obser1, temp1 + dither(temp2, delay))
		loss += L1loss(obser2, temp2 + dither(temp1, -delay))
		loss /=2

	temp1 = myimtocol(temp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
	temp2 = myimtocol(temp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
	out = np.concatenate((temp1,temp2),axis=0)
	B,H,W = out.shape
	out = out.reshape(B,1,H,W)
	outloss = np.zeros((1,1,config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),dtype='float32')
	outloss[:,:,0,0] = loss
	if flag1 and not test:
		ntemp1 = obser1 - dither(d2de1, delay)*np.random.uniform(0.75, 0.91)
		ntemp2 = obser2 - dither(d1de1, -delay)*np.random.uniform(0.75, 0.91)
		ntemp1 = myimtocol(ntemp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
		ntemp2 = myimtocol(ntemp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
		out2 = np.concatenate((ntemp1, ntemp2), axis=0)
		B, H, W = out2.shape
		out2 = out2.reshape(B, 1, H, W)
		out = np.concatenate((out, out2), axis=0)
	if flag2 and not test:
		out3 = copy.deepcopy(out)
		for i in range(out3.shape[0]):
			out3[i, :, :, :] = out3[i, :, :, :]*np.random.uniform(0.95, 1.05)
		out = np.concatenate((out, out3), axis=0)
		out = out[:264, :, :, :]
	B, C, H, W = out.shape
	avg = B//4
	new = np.concatenate((out[:avg,:,:,:], outloss,out[avg:avg*2,:,:,:], outloss,out[avg*2:avg*3,:,:,:], outloss,out[avg*3:,:,:,:], outloss), axis=0)
	return torch.from_numpy(new)

def L1loss(X,Y):
	diff = np.add(X, -Y)
	error = np.sqrt(diff * diff + 1e-6)
	loss = np.mean(error)
	return loss

