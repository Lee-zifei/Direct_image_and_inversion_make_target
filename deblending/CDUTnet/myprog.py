import numpy as np
import math

# convert 2D matrix to patches or invert
def myimtocol(input1,rn1,rn2,n1,n2,tslide,xslide,f):
    if f == 1:
        n1,n2 = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        datasize = num1 * num2
        output1 = np.zeros((datasize, rn1, rn2), dtype='float32')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 1):
                    if (j < num1 - 1):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, i * xslide:i * xslide + rn2];
                else:
                    if (j < num1 - 1):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, n2 - rn2:n2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, n2 - rn2:n2];
    else:
        [datasize, rn1, rn2] = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        output1 = np.zeros((n1,n2), dtype='float32')
        weight = np.zeros((n1,n2), dtype='float32')
        one = np.ones((rn1,rn2), dtype='float32')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 1):
                    if (j < num1 - 1):
                        output1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] += one;
                    else:
                        output1[n1 - rn1:n1, i * xslide:i * xslide + rn2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, i * xslide:i * xslide + rn2] += one;
                else:
                    if (j < num1 - 1):
                        output1[j * tslide:j * tslide + rn1, n2 - rn2:n2] +=  np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, n2 - rn2:n2] += one;
                    else:
                        output1[n1 - rn1:n1, n2 - rn2:n2] += np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, n2 - rn2:n2] +=  one;

        output1 = output1/weight
    return output1



def myimtocol2(input1,rn1,rn2,n1,n2, n3,tslide,xslide,f):
    if f == 1:
        num1 = int(np.floor((n2 - rn1) / tslide) + 1 + (np.mod(n2 - rn1, tslide) != 0))
        num2 = int(np.floor((n3 - rn2) / xslide) + 1 + (np.mod(n3 - rn2, xslide) != 0))
        output1 = np.zeros((n1*num1*num2,rn1,rn2), dtype='float32')
        for i in range (n1):
            temp = input1[i,:,:]
            if i == 0:
                d2patch = myimtocol(temp, rn1,rn2,n2, n3,tslide,xslide,1)
                output1= d2patch
            else:
                d2patch = myimtocol(temp, rn1,rn2,n2, n3,tslide,xslide,1)
                output1= np.concatenate((output1,d2patch), axis=0)

    else:
        output1= np.zeros((n2,n1*n3), dtype='float32')
        num1 = int(np.floor((n2 - rn1) / tslide) + 1 + (np.mod(n2 - rn1, tslide) != 0))
        num2 = int(np.floor((n3 - rn2) / xslide) + 1 + (np.mod(n3 - rn2, xslide) != 0))
        for i in range (n1):
            temp = input1[i*num1*num2:(i+1)*num1*num2,:,:]
            if i == 0:
                d2patch = myimtocol(temp,rn1,rn2,n2,n3,tslide,xslide,0)
                output1 = d2patch
            else:
                d2patch = myimtocol(temp,rn1,rn2,n2,n3,tslide,xslide,0)
                output1= np.concatenate((output1,d2patch), axis=1)
    return output1




def mutter(inputs, x0, t0, k):
    nt,nx=inputs.shape[0],inputs.shape[1]
    output=inputs
    for i in range(1,nt):###axis y
        for j in range(1,nx):###axis x
            if j < x0 :
                if i < math.floor(-k*j+t0+k*x0):
                    output[i,j]=0
            if j >=x0:
                if i < math.floor(k*j+t0-k*x0):
                    output[i,j]=0
    return output

def turnone(inputs):####数据归一化
    vmax=np.max(np.abs(inputs))
    vmin=np.min(np.abs(inputs))
    outputs=(inputs-vmin)/(vmax-vmin)
    return outputs




# dither time  time/unit
def dither(input1, time):
    [n1, n2] = input1.shape
    out1 = np.zeros((n1,n2), dtype='float32')
    n22 = len(time)
    if n2 != n22:
        print ('Error in size of delay time')
    for ix in range(n2):
        for it in range(n1):
            itt = it + int(time[ix])
            if itt >= 0 and itt < n1 :
                out1[itt,ix] = input1[it,ix]
    return out1
def itpdither(input1,time,listn):
    [n1, n2] = input1.shape
    out1 = np.zeros((n1,n2), dtype='float32')
    n22 = len(time)
    if n2 != n22:
        print ('Error in size of delay time')
    for ix in range(n2):
        if listn[ix]==1:
            continue
        for it in range(n1):
            itt = it + int(time[ix])
            if itt >= 0 and itt < n1 :
                out1[itt,ix] = input1[it,ix]
    return out1

def generate(n2,per,mode=None):
    if mode==None:
        x1=np.random.randint(1,4,1)
    if mode==0:
        x1=np.random.randint(1,3,1)
    if x1==1:##一个
        x2=np.random.randint(5,7,1)
        a1=np.random.randint(7,n2-int(x2),1)
        te=np.zeros((int(x2),), dtype='int64')
        for i in range(int(x2)):
            te[i]=int(a1)+i
        if int(a1) <= 0.3*n2 or int(a1) >= 0.7*n2 :
            a2=np.random.randint(1,int((int(per*n2)-int(x2))*0.3),1)###左边
            a3=int(per*n2)-int(a2)-int(x2)+2
        else:
            a2=np.random.randint(1,int(per*n2)-int(x2),1)####左边几个
            a3=int(per*n2)-int(a2)-int(x2)+2###右边几个
        t1=np.random.randint(1,int(a1),int(a2))
        t2=np.random.randint(int(a1)+int(x2),n2,a3)
        temp=np.concatenate((te,t1,t2),axis=0)
    if x1==3:
        temp=np.random.randint(1,n2,int(n2*per))
    if x1==2:####两个
        x2=np.random.randint(5,7,1)
        x3=np.random.randint(5,7,1)
        a1=np.random.randint(10,int(n2*0.4)-int(x2),1)###小
        a2=np.random.randint(int(n2*0.4),n2-int(x3),1)###大
        te1=np.zeros((int(x2),), dtype='int64')
        te2=np.zeros((int(x3),), dtype='int64')
        for i in range(int(x2)):
            te1[i]=int(a1)+i
        for i in range(int(x3)):
            te2[i]=int(a2)+i
        if int(a1) <= 0.4*n2  :
            a3=np.random.randint(3,int((int(per*n2)-int(x2)-int(x3)-6)*0.4),1)
            a4=np.random.randint(3,int(per*n2)-int(x2)-int(x3)-int(a3),1)##right
        if int(a2) >= 0.6*n2 :
            a3=np.random.randint(2,int(per*n2)-int(x2)-int(x3)-6,1)
            a4=np.random.randint(1,int((int(per*n2)-int(x2)-int(x3)-int(a3))*0.4),1)#right
        else:
            a3=np.random.randint(3,int(per*n2)-int(x2)-int(x3)-7,1)
            a4=np.random.randint(3,int(per*n2)-int(x2)-int(x3)-int(a3),1)###左
        a5=int(per*n2)-int(a3)-int(a4)-int(x2)-int(x3)+4###mid
        t1=np.random.randint(1,int(a1),int(a3))
        t2=np.random.randint(int(a2),n2,int(a4))
        t3=np.random.randint(int(a1),int(a2),int(a5))
        temp=np.concatenate((t1,t2,t3,te1,te2),axis=None)
    return temp

def generatesimple(n2,per,mode=None):
    if mode==None:
        x1=np.random.randint(1,4,1)
    else:
        x1=np.random.randint(2,4,1)
    if x1==1:##一个
        x2=np.random.randint(5,7,1)
        a1=np.random.randint(5,n2-int(x2),1)
        te=np.zeros((int(x2),), dtype='int64')
        for i in range(int(x2)):
            te[i]=int(a1)+i+1
        aa=int(per*n2)-int(x2)
        t1=np.random.randint(1,n2,aa+int(int(x2)/int(per*n2)*aa/2))
        temp=np.concatenate((te,t1),axis=0)
    if x1==2:####两个
        x2=np.random.randint(5,7,1)
        x3=np.random.randint(5,7,1)
        a1=np.random.randint(6,int(n2*0.4)-int(x2),1)###小
        a2=np.random.randint(int(n2*0.4),n2-int(x3),1)###大
        te1=np.zeros((int(x2),), dtype='int64')
        te2=np.zeros((int(x3),), dtype='int64')
        for i in range(int(x2)):
            te1[i]=int(a1)+i+1
        for i in range(int(x3)):
            te2[i]=int(a2)+i+1
        aa=int(per*n2)-int(x2)-int(x3)
        t1=np.random.randint(1,n2,aa+int((int(x2)+int(x3))/int(per*n2)*aa/2))
        temp=np.concatenate((t1,te1,te2),axis=0)
    if x1==3:####三个
        x2=np.random.randint(5,7,1)
        x3=np.random.randint(5,7,1)
        x4=np.random.randint(5,7,1)
        a1=np.random.randint(1,int(n2*0.3)-int(x2),1)
        a2=np.random.randint(int(n2*0.3),int(n2*0.6)-int(x3),1)
        a3=np.random.randint(int(n2*0.6),n2-int(x4),1)
        te1=np.zeros((int(x2),), dtype='int64')
        te2=np.zeros((int(x3),), dtype='int64')
        te3=np.zeros((int(x4),), dtype='int64')
        for i in range(int(x2)):
            te1[i]=int(a1)+i+1
        for i in range(int(x3)):
            te2[i]=int(a2)+i+1
        for i in range(int(x4)):
            te3[i]=int(a3)+i+1
        aa=int(per*n2)-int(x2)-int(x3)-int(x4)
        t1=np.random.randint(1,n2,aa+int((int(x2)+int(x3)+int(x4))/int(per*n2)*aa/2))
        temp=np.concatenate((t1,te1,te2,te3),axis=0)
    return temp
# calculate signal-to-noise ratio
def calculate_snr(signal,noisy):
    error = signal - noisy
    temp1 = np.sum(np.sum(signal*signal))
    temp2 = np.sum(np.sum(error*error))
    snr = 10*np.log10(temp1/temp2)
    return snr
