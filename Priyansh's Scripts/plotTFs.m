function plotCTFs
% Fig 2c in Foster et al. Psych Science

% specify subjects to plot
subs = [2];%,2];%,3,7,8,9,10,11,12,14,15,16,17,18,19,20];
nSubs = length(subs);

root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'/Work Study/IEM/data/'];

% specify filename of data to plot
name = '_SpatialTF.mat';

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
    %fName = [dRoot,num2str(sn),name];
    %load(fName)
    em = py.pickle.load(py.open(['./data/', num2str(i), '_SpatialTF_.pickle'], 'rb'));
    %fois = ismember(em.bands,'Alpha');
    tfs_total = double(em{'tfs.total'});
    t_ctf(i,:,:) = squeeze(mean(mean(tfs_total(1,1:10,:,:,:),4),2)); % grab data, average across iterations (dimension 2) and test blocks (dimension 4). 
end


% average data across subjects, mirror the -180 channel at +180
tctf = squeeze(mean(t_ctf,1));
tctf = [tctf tctf(:,1)];
fprintf(['Size of tctf ' num2str(size(tctf)) '\n'])


% Setup plot axes
x = linspace(-180,180,nChans+1);
nTimes = size(tctf,1); nBins = length(x);
X = repmat(x',1,nTimes);

fprintf(['Size of double(em{time}) ' num2str(size(double(em{'time'}))) '\n'])
fprintf(['double(nBins) is ' num2str(double(em{'nBins'})) '\n'])

Y = repmat(double(em{'time'}),nBins,1);

fprintf(['Size of X ' num2str(size(X)) '\n'])
fprintf(['Size of Y ' num2str(size(Y)) '\n'])

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
em_time = double(em{'time'});
axis([x(1) x(end) em_time(1) em_time(end) lim]);
set(gca,'XTick',[-180:90:180])
set(gca,'YTick',[-500:500:2000])
% title('Total Power','FontSize',ts)
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