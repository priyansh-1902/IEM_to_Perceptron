function plotCTFs
% Fig 2c in Foster et al. Psych Science

% specify subjects to plot
subs = [3];%[1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20];
nSubs = length(subs);

root = pwd; out = 'AnalysisScripts';
%dRoot = [root(1:end-length(out)),'sity of Toronto/IEM/data/'];

% specify filename of data to plot
name = '_PYPerceptronTF.mat';

% Plot specs
nChans = 8; % number of location-selective channels
view3d = [90 90]; % view angle of 3d plots
ts = 15; % title size
fs = 9; % font size
lim = [0 0.6]; % limits on color map and channel response axis
times = -500:4:1248;
markerPos = -165; % position of the significance marker

root = pwd;

% loop through participants and grab tf generated by forward model
for i = 1:nSubs
    sn = subs(i);
    fName = [num2str(sn) name];
    load(fName);
    %fois = ismember(em.bands,'Alpha');
    %t_ctf(i,:,:) = squeeze(mean(mean(em.tfs.total(fois,1:10,:,:,:),4),2)); % grab data, average across iterations (dimension 2) and test blocks (dimension 4). 
    t_ctf(i, :, :) = squeeze(mean(mean(tfs(1:10, :, :, :),1),3));
end


% average data across subjects, mirror the -180 channel at +180
tctf = squeeze(mean(t_ctf,1)); tctf = [tctf tctf(:,1)];
% Setup plot axes
x = linspace(-180,180,nChans+1);
nTimes = size(tctf,1); nBins = length(x);
X = repmat(x',1,nTimes);
Y = repmat(times,nBins,1);

% Plot time-resolved total tuning function
FigHandle = figure('Position', [100, 100, 480, 180]); % size of the plot
surf(X,Y,tctf','EdgeColor','none','LineStyle','none','FaceLighting','phong')
shading interp
h=findobj('type','patch');
set(h,'linewidth',2)
hold on
set(gca, 'box','off')
set(gca,'color','none')
set(gca,'LineWidth',1,'TickDir','out');
set(gca,'FontSize',fs)
set(gca,'FontName','Arial Narrow')
view(view3d)
axis([x(1) x(end) times(1) times(end) lim]);
set(gca,'XTick',[-180:90:180])
set(gca,'YTick',[-500:100:2000])
title('Linear Perceptron Model','FontSize',ts)
xlabel({'Channel Offset'});
ylabel('Time (ms)');
set(get(gca,'xlabel'),'rotation',90); %where angle is in degrees
caxis([lim])
grid off
c = colorbar;
ax = gca;
axpos = ax.Position;
cpos = c.Position;
cpos(3) = 0.3*cpos(3);
c.Position = cpos;
ax.Position = axpos;
ylabel(c, 'Channel Response');