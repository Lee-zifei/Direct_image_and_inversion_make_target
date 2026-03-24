import torch
import torch.nn as nn
from functools import partial

class center(nn.Module):
	def __init__(self,  dim=64,bias=False,norm_layer=partial(nn.BatchNorm2d, eps=1e-3)):
		super(residual_block, self).__init__()
		self.layer1=nn.Conv2d(dim,dim,3,stride=1,padding=1,bias=bias)
		self.ac1=nn.ReLU()
		self.layer2=nn.Conv2d(dim,dim,3,stride=1,padding=1,bias=bias)
		self.ac2=nn.ReLU()
		self.layer3=nn.Conv2d(dim,dim,3,stride=1,padding=1,bias=bias)
		self.ac3=nn.ReLU()
	def forward(self,x):
		x=self.layer1(x)
		x=self.ac1(x)
		x=self.layer2(x)
		x=self.ac2(x)
		x=self.layer3(x)
		x=self.ac3(x)
		return x

class residual_block(nn.Module):
	def __init__(self,  dim=64,bias=False,norm_layer=partial(nn.BatchNorm2d, eps=1e-3)):
		super(residual_block, self).__init__()
		med_planes=dim//4
		self.layer1=nn.Conv2d(dim,med_planes,1,stride=1,padding=1,bias=bias)
		self.ac1=nn.ReLU()
		self.layer2=nn.Conv2d(med_planes,med_planes,3,stride=1,padding=1,bias=bias)
		self.ac2=nn.ReLU()
		self.layer3=nn.Conv2d(med_planes,dim,1,stride=1,padding=1,bias=bias)
		self.ac3=nn.ReLU()

	def forward(self,x):
		shorcut=x
		x=self.layer1(x)
		x=self.ac1(x)
		x=self.layer2(x)
		x=self.ac2(x)
		x=self.layer3(x)
		x=self.ac3(x)
		x+=shorcut
		return x
class NakaUnet(nn.Module):
	def __init__(self,  image_channels=1, kernel_size=3,norm_layer=partial(nn.BatchNorm2d, eps=1e-3),bias=False):
		super(NakaUnet, self).__init__()
		dim1=16
		dim2=64
		dim3=128

		self.conv1=nn.Conv2d(image_channels,dim2,kernel_size,stride=1,padding=1,bias=bias)
		self.act1=nn.ReLU()
		self.residual1=residual_block()

		self.down1=nn.MaxPool2d(2)

		self.conv2=nn.Conv2d(dim2,dim2,kernel_size,stride=1,padding=1,bias=bias)
		self.act2 = nn.ReLU()
		self.residual2 = residual_block()

		self.down2=nn.MaxPool2d(2)

		self.conv3=nn.Conv2d(dim2,dim2,kernel_size,stride=1,padding=1,bias=bias)
		self.act3 = nn.ReLU()
		self.residual3 = residual_block()

		self.down3 = nn.MaxPool2d(2)
		self.conv4=nn.Conv2d(dim2,dim2,kernel_size,stride=1,padding=1,bias=bias)
		self.act4 = nn.ReLU()
		self.residual4 = residual_block()


		self.down4 = nn.MaxPool2d(2)
		self.center=center()

		self.up1=nn.ConvTranspose2d(dim3,dim2,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.reduce1=nn.Conv2d(dim3, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.actr1 = nn.ReLU()
		self.residualu1=residual_block()

		self.up2=nn.ConvTranspose2d(dim3,dim2,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.reduce2=nn.Conv2d(dim3, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.actr2 = nn.ReLU()
		self.residualu2=residual_block()

		self.up3=nn.ConvTranspose2d(dim3,dim2,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.reduce3=nn.Conv2d(dim3, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.actr3 = nn.ReLU()
		self.residualu3=residual_block()

		self.up4=nn.ConvTranspose2d(dim3,dim2,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.reduce4=nn.Conv2d(dim3, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.actr4 = nn.ReLU()

		self.en1=nn.Conv2d(dim2, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.enbn1=norm_layer(dim2)
		self.enact1 = nn.ReLU()
		self.output=nn.Conv2d(dim2, image_channels , kernel_size, stride=1, padding=1, bias=bias)

		self.apply(self._initialize_weights)

	def forward(self,x):

		s1=self.conv1(x)
		s1=self.act1(s1)
		s1=self.residual1(s1)

		d1=self.down(s1)

		s2=self.conv2(d1)
		s2=self.act2(s2)
		s2=self.residual2(s2)

		d2=self.down(s2)

		s3=self.conv3(d2)
		s3=self.act3(s3)
		s3=self.residual3(s3)

		d3=self.down(s3)

		s4 = self.conv4(d3)
		s4 = self.act4(s4)
		s4 = self.residual4(s4)

		d4 = self.down(s4)

		center=self.center(d4)

		u1=self.up1(center)
		cat1=torch.cat([u1,s4], 1)
		e1=self.reduce1(cat1)
		e1=self.actr1(e1)
		e1=self.residualu1(e1)


		u2=self.up2(e1)
		cat2=torch.cat([u2,s3], 1)
		e2=self.reduce2(cat2)
		e2=self.actr2(e2)
		e2=self.residualu2(e2)

		u3=self.up1(e2)
		cat3=torch.cat([u3,s2], 1)
		e3=self.reduce1(cat3)
		e3=self.actr3(e3)
		e3=self.residualu3(e3)

		u4=self.up1(e3)
		cat4=torch.cat([u4,s1], 1)
		e4=self.reduce4(cat4)
		e4=self.actr4(e4)

		ef=self.en1(e4)
		ef=self.enact1(ef)
		ef=self.output(ef)
		
		return ef