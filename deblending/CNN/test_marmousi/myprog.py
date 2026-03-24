import numpy as np


# convert 2D matrix to patches or invert
def myimtocol(input1,rn1,rn2,n1,n2,tslide,xslide,f):
    if f == 1:
        n1,n2 = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        datasize = num1 * num2
        output1 = np.zeros((datasize, rn1, rn2), dtype='float')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 2):
                    if (j < num1 - 2):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, i * xslide:i * xslide + rn2];
                else:
                    if (j < num1 - 2):
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, n2 - rn2:n2];
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, n2 - rn2:n2];
    else:
        [datasize, rn1, rn2] = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        output1 = np.zeros((n1,n2), dtype='float')
        weight = np.zeros((n1,n2), dtype='float')
        one = np.ones((rn1,rn2), dtype='float')

        for i in range(num2):
            for j in range(num1):
                if (i < num2 - 2):
                    if (j < num1 - 2):
                        output1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] = output1[j * tslide:j * tslide + rn1,i * xslide:i * xslide + rn2] + np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] = weight[j * tslide:j * tslide + rn1,i * xslide:i * xslide + rn2] + one;
                    else:
                        output1[n1 - rn1:n1, i * xslide:i * xslide + rn2] = output1[n1 - rn1:n1,i * xslide:i * xslide + rn2] + np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, i * xslide:i * xslide + rn2] = weight[n1 - rn1:n1,i * xslide:i * xslide + rn2] + one;
                else:
                    if (j < num1 - 2):
                        output1[j * tslide:j * tslide + rn1, n2 - rn2:n2] = output1[j * tslide:j * tslide + rn1,n2 - rn2:n2] + np.squeeze(input1[i * num1 + j, :, :]);
                        weight[j * tslide:j * tslide + rn1, n2 - rn2:n2] = weight[j * tslide:j * tslide + rn1,n2 - rn2:n2] + one;
                    else:
                        output1[n1 - rn1:n1, n2 - rn2:n2] = output1[n1 - rn1:n1, n2 - rn2:n2] + np.squeeze(input1[i * num1 + j, :, :]);
                        weight[n1 - rn1:n1, n2 - rn2:n2] = weight[n1 - rn1:n1, n2 - rn2:n2] + one;

        output1 = output1/weight
    return output1


# dither time  time/unit
def dither(input1, time):
    [n1, n2] = input1.shape
    out1 = np.zeros((n1,n2), dtype='float')
    n22 = len(time)
    if n2 != n22:
        print ('Error in size of delay time')
    for ix in range(n2):
        for it in range(n1):
            itt = it + int(time[ix])
            if itt >= 0 and itt < n1 :
                out1[itt,ix] = input1[it,ix]
    return out1




# calculate signal-to-noise ratio
def calculate_snr(signal,noisy):
    error = signal - noisy
    temp1 = np.sum(np.sum(signal*signal))
    temp2 = np.sum(np.sum(error*error))
    snr = 10*np.log10(temp1/temp2)
    return snr
