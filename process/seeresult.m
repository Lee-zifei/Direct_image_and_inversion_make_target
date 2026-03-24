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
n1 = 3000;
n2 = 559;

clip = 0.3;
mm = seis(2);

stack = zread('stack.dat',[n1,n2]);
stack = stack./max(max(stack));
stack_kif = zread('stack_kif.dat',[n1,n2]);
stack_kif = stack_kif./max(max(stack_kif));

zfig([stack,stack_kif],clip,mm);