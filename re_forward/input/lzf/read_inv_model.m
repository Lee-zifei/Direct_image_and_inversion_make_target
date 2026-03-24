clear;clc;close all;delete(gcp('nocreate'))  
%Linux
if exist('/media/lzf/Work/code/matlab/mat_toolbox/myprogs','dir')
    addpath('/media/lzf/Work/code/matlab/mat_toolbox/myprogs');
    addpath('/media/lzf/Work/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/media/lzf/Work/code/matlab/mat_toolbox/crewes'));
    datapath='/media/lzf/Work/data'; 
elseif exist('L:\code\matlab\mat_toolbox\myprogs','dir')
%Windows
    addpath('L:\code\matlab\mat_toolbox\myprogs');
    addpath('L:\code\matlab\mat_toolbox\CurveLab-2.1.3\fdct_wrapping_matlab');
    addpath(genpath('L:\code\matlab\mat_toolbox\crewes'));
    datapath='L:\data'; 
elseif exist('/data/data1/lzf/code/matlab/mat_toolbox/myprogs','dir')
%Server
    addpath('/data/data1/lzf/code/matlab/mat_toolbox/myprogs');
    addpath('/data/data1/lzf/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/data/data1/lzf/code/matlab/mat_toolbox/rewes'));
    datapath='/data/data1/lzf/data'; 
else
%MAC
    addpath('/Users/zifeilee/code/matlab/mat_toolbox/myprogs');
    addpath('/Users/zifeilee/code/matlab/mat_toolbox/CurveLab-2.1.3/fdct_wrapping_matlab');
    addpath(genpath('/Users/zifeilee/code/matlab/mat_toolbox/crewes'));
end
%###################################################################################################
clear;clc;close all;
[Data1,SegyTraceHeaders,SegyHeader]=ReadSegy('./Model.sgy');
% [Data2,SegyTraceHeaders,SegyHeader]=ReadSegy('./lzf-PQR_Q.sgy');
% [Data3,SegyTraceHeaders,SegyHeader]=ReadSegy('./lzf-PQR.sgy');
figure;
imagesc(Data1);
Data1(1,:) = [];
Data1(:,1) = [];
zsave('../inversion_marmousi.dat',Data1);
% figure;
% imagesc(Data2);
% figure;
% imagesc(Data3);