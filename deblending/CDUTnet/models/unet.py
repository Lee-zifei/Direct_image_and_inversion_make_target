import torch
import torch.nn as nn


class Unet(nn.Module):
	def __init__(self,  image_channels=1, kernel_size=3,bias=True):
		dim1=32
		dim2=64
		dim3=128
		super(Unet, self).__init__()

		self.conv1=nn.Conv2d(image_channels,dim1,kernel_size,stride=1,padding=1,bias=bias)
		self.act1=nn.ReLU()
		self.conv2=nn.Conv2d(dim1,dim1,kernel_size,stride=1,padding=1,bias=bias)
		self.act2= nn.ReLU()

		self.down1=nn.MaxPool2d(2)
		self.conv3=nn.Conv2d(dim1,dim2,kernel_size,stride=1,padding=1,bias=bias)
		self.act3 = nn.ReLU()
		self.conv4 = nn.Conv2d(dim2, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.act4 = nn.ReLU()

		self.down2=nn.MaxPool2d(2)
		self.conv5= nn.Conv2d(dim2,dim3,kernel_size,stride=1,padding=1,bias=bias)
		self.act5 = nn.ReLU()
		self.conv6 = nn.Conv2d(dim3, dim3, kernel_size, stride=1, padding=1, bias=bias)
		self.act6 = nn.ReLU()

		self.up1=nn.ConvTranspose2d(dim3,dim2,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.act7 = nn.ReLU()
		self.reduce1=nn.Conv2d(dim3, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.act8 = nn.ReLU()
		self.conv7=nn.Conv2d(dim2, dim2, kernel_size, stride=1, padding=1, bias=bias)
		self.act9 = nn.ReLU()

		self.up2=nn.ConvTranspose2d(dim2,dim1,kernel_size,stride=2,padding=1,output_padding=1,bias=bias)
		self.act10 = nn.ReLU()
		self.reduce2=nn.Conv2d(dim2, dim1, kernel_size, stride=1, padding=1, bias=bias)
		self.act11 = nn.ReLU()
		self.conv8=nn.Conv2d(dim1, image_channels , kernel_size, stride=1, padding=1, bias=bias)

	def forward(self,x):

		c1=self.conv1(x)
		c1=self.act1(c1)

		c2=self.conv2(c1)
		c2=self.act2(c2)

		d1=self.down1(c2)

		c3=self.conv3(d1)
		c3=self.act3(c3)

		c4=self.conv4(c3)
		c4=self.act4(c4)

		d2=self.down2(c4)

		c5=self.conv5(d2)
		c5=self.act5(c5)

		c6=self.conv6(c5)
		c6=self.act6(c6)

		u1=self.up1(c6)
		u1=self.act7(u1)

		cat1=torch.cat([u1,c4], 1)

		re1=self.reduce1(cat1)
		re1=self.act8(re1)

		c7=self.conv7(re1)
		c7=self.act9(c7)

		u2=self.up2(c7)
		u2=self.act10(u2)

		cat2=torch.cat([u2,c2], 1)

		re2=self.reduce2(cat2)
		re2=self.act11(re2)
		c8=self.conv8(re2)

		return c8
