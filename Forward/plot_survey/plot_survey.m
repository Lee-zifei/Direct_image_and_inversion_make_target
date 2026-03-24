clear;clc; close all
data= {};
fid = fopen("./parameter.txt","r");
while ~feof(fid)  % 当没有到达文件末尾时继续循环
    rline = fgetl(fid);  % 读取一行文本
    data = [data; rline];  % 将行添加到数据数组中
end
fclose(fid);

%% read parameterss from parameters.txt 
parameters.nx=str2num(data{2});
parameters.nz=str2num(data{4});
parameters.pml=str2num(data{6});
parameters.lc=str2num(data{8});
parameters.dx=str2num(data{10});
parameters.dz=str2num(data{12});
parameters.rectime=str2num(data{14});
parameters.dt=str2num(data{16});
parameters.f0=str2num(data{18});
parameters.ns=str2num(data{20});
parameters.sx0=str2num(data{22});
parameters.shotdx=str2num(data{24});
parameters.shotdep=str2num(data{26});
parameters.r_n=str2num(data{28});
parameters.rx0=str2num(data{30});
parameters.recdx=str2num(data{32});
parameters.recdep=str2num(data{34});
parameters.simu=str2num(data{36});
parameters.s_distance=str2num(data{38});


%% read f
% vfile= './input/test1_1_1.dat';
vfile= '../input/mar_234_663.dat';

fid = fopen(vfile, "r");
vel =fread(fid,[parameters.nz parameters.nx],'float');
fclose(fid);
vmin = min(min(vel));
vmax = max(max(vel));

figure;imagesc(vel,[vmin vmax]);
% 
screenratio= parameters.nx/parameters.nz;
set(gca,'Position',[0.1 0.13 0.8 0.75], 'fontname','times new roman', 'xaxislocation','bottom', 'yaxislocation','left');
set(gcf,'Units','centimeters','Position',[5 5 15*screenratio 15 ]);
set(gca,'xtick',200:200:1000);
set(gca,'xticklabel',1:5)
set(gca,'ytick',100:100:200);
set(gca,'yticklabel',0.5:0.5:2)
set(gca,'fontsize',20);
ylabel('Depth (km)','Fontsize',20);
xlabel('Distance (km)','Fontsize',20);

kk=5;
zrec = parameters.recdep;
xrec = floor(parameters.rx0/parameters.dx)+1:kk*parameters.recdx/parameters.dx:(parameters.rx0+ parameters.recdx*parameters.r_n)/parameters.dx;
line(xrec,zrec,'color','r','linestyle','none','marker','V','linewidth',1,'markersize',8)

zshot = parameters.shotdep*2;
xshot = floor(parameters.sx0/parameters.dx)/+1:kk*parameters.shotdx/parameters.dx:(parameters.rx0+ parameters.shotdx*parameters.ns)/parameters.dx;
line(xshot,zshot,'color','g','linestyle','none','marker','*','linewidth',1,'markersize',8)
% 
if parameters.simu==1
    zshot2 = parameters.shotdep*3;
    xshot2 = floor((parameters.sx0+parameters.s_distance)/parameters.dx)+1:kk*parameters.shotdx/parameters.dx: ...
        (parameters.sx0+parameters.s_distance+ parameters.shotdx*parameters.ns)/parameters.dx;
line(xshot2,zshot2,'color','k','linestyle','none','marker','*','linewidth',1,'markersize',8)
end
