clear;clc; close all
data= {};
fid = fopen("../parameter.txt","r");
while ~feof(fid)  % 当没有到达文件末尾时继续循环
    rline = fgetl(fid);  % 读取一行文本
    data = [data; rline];  % 将行添加到数据数组中
end
fclose(fid);

%% read parameter from parameter.txt 
parameter.nx=str2num(data{2});
parameter.nz=str2num(data{4});
parameter.pml=str2num(data{6});
parameter.lc=str2num(data{8});
parameter.dx=str2num(data{10});
parameter.dz=str2num(data{12});
parameter.rectime=str2num(data{14});
parameter.dt=str2num(data{16});
parameter.f0=str2num(data{18});
parameter.ns=str2num(data{20});
parameter.sx0=str2num(data{22});
parameter.shotdx=str2num(data{24});
parameter.shotdep=str2num(data{26});
parameter.r_n=str2num(data{28});
parameter.rx0=str2num(data{30});
parameter.recdx=str2num(data{32});
parameter.recdep=str2num(data{34});
parameter.simu=str2num(data{36});
parameter.s_distance=str2num(data{38});





% 
randn('seed',20240830)


a=randn(1,parameter.ns);
delay1=abs(0.3*a./abs(max(a)));
delay1=3*delay1 .* (delay1<0.25);

delay1 = floor(delay1/parameter.dt)*parameter.dt;

%figure;plot(delay1)

filename = 'fshot_dela.dat';
fid = fopen(filename,'w');
fwrite(fid,delay1,'float');
fclose(fid);

parameter.ns

maxdit = max(delay1/parameter.dt);
printf ('Max dither is %d\n',maxdit)
mindit = min(delay1/parameter.dt);
printf ('Min dither is %d\n',mindit)

sum(delay1)/parameter.ns/parameter.dt

