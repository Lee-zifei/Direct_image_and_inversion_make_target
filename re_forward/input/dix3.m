close all; clc; clear;
addpath('/home/chen/Documents/Forward/Forward_RTM');
filename = 'chj-PQR.sgy';
[traces, segyHeader] = ReadSegy(filename);
% snap = zread(filename);
% snap = reshape(snap,[3000 ,767]);
% traces =snap;
vel = traces;   
[nz, nx] = size(vel);
dz = 10;
delta_t = 0.0005;  
nt=3000;
t_idx = zeros(nz,nx);
vrms_time = zeros(nt,nx);
for i = 1:nx
    v_col = vel(:, i);      
    v2 = v_col.^2;
    dt = dz ./ v_col;     
    cum_t = cumsum(dt);
    t_idx(:,i) = cum_t;
    %%%%Dix  只做时深转换的话不需要这一步
    cum_v2dt = cumsum(v2 .* dt);
    vrms_col = sqrt(cum_v2dt ./ cum_t);   
    vrms_map(:,i)=vrms_col;
    %%%%
end
for i = 1:nx
    for j =1:nz
t = ceil(t_idx(j,i)/delta_t);
vrms_time(t,i)=vrms_map(j,i);%%只做时深转换将vrms_map 替换为vel
    end
end
   figure;
imagesc(vrms_time); 
colorbar; 
for i = 1:nx
    for j = 2:nt
        if vrms_time(j,i) ==0
            vrms_time(j,i) = vrms_time(j-1,i);
        end
    end
end
for i = 1:nx
    for j = 1:nt
        if vrms_time(j,i) ==0
            vrms_time(j,i) =1500;
        end
    end
end



    

figure;
imagesc(vel); 
colorbar; 
title('RMS-depth ');


figure;
imagesc(vrms_time); 
colorbar; 
title('RMS-time');
filename = 'salt3.dat';
fid = fopen(filename,'w');
fwrite(fid,vel, 'float');
fclose(fid);




