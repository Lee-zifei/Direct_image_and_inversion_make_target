import numpy as np
import torch
import os
from concurrent.futures import ProcessPoolExecutor

def parallbst(a,index,corenum=8):

	a=a.cpu().numpy()
	index=index.cpu().numpy()

	outs=np.zeros((a.shape),dtype='float32')

	p = ProcessPoolExecutor() ###多进程
	l = []
	results = []
	num=corenum//a.shape[0] ##每个核拿多少个batch
	for i in range(corenum):
		l.append(p.submit(invbst,a,index,[x for x in range(i,i+num)]))

	p.shutdown()
	for obj in l:
		results.append(obj.result())  ##### tuple   [batch_number,seislet_transform]

	for i in range(corenum):
		batch_number, out=results[i]
		for j,num in enumerate(batch_number):
			outs[num,:,:,:] = out[j,:,:,:]

	outst=torch.from_numpy(outs).cuda()
	return outst

def invbst(a,index,batch_number): #### B C H W
	#### a->torch.cuda
	for i in batch_number:
		out = np.zeros((a.shape), dtype='float32')
		hyper=a[i,0,:,:].T
		name=str(index[i,0])
		hyper=np.ascontiguousarray(hyper,dtype='float32')
		with open(name+'_pseislet.dat','wb') as f: ### 网络处理后输出成dat
			f.write(hyper)

		invprocess2dat(name,hyper.shape[1],hyper.shape[0]) ### 从seislet域转为时间域
		datas = dat2numpy(name + "_denoise", size=(hyper.shape[1],hyper.shape[0])) ## 从dat转成numpy
		os.system('''rm %s_denoise.dat''' % (name))
		datas = np.reshape(datas,  (1,)+datas.shape )
		out[i,:,:,:] = datas

	return batch_number,out

def dat2numpy(name,size=(128,128)):
	img=np.fromfile(name+'.dat',dtype='float32')
	img=img.reshape(size[0],size[1]).T
	return img

def invprocess2dat(name,n1,n2,d1=0.004,d2=1,o1=0,o2=0):
	os.system('''echo n1=%i n2=%i d1=%g d2=%g o1=%g o2=%g data_format="native_float" esize=4 in=%s_pseislet.dat > %s_pseislet.rsf'''%(n1, n2, d1, d2, o1, o2, name, name))
	os.system('''sfseislet < %s_pseislet.rsf > %s_denoise.rsf dip=%s_dip.rsf inv=y adj=n eps=1 unit=n order=1 type=b'''%(name, name, name))
	os.system('''< %s_denoise.rsf sfdd type=float | sfrsf2bin > %s_denoise.dat'''%(name,name))
	os.system('''rm %s_pseislet.rsf'''%(name))
	os.system('''rm %s_denoise.rsf''' % (name))
	os.system('''rm %s_pseislet.dat''' % (name))

