function plotAveCTF_Delta

% Fig 2f (right panel) in Foster et al. Psych Science

set(0,'DefaultAxesFontName', 'HelveticaNeue LT 57 Cn')
fs = 9; % size for axis numbers
labelSize = 10; % size for axis labels


% specify some details
times = -500:4:1248;
nChans = 8;

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

% load data file
fName = [dRoot,'bootstrappedCTFs_delta.mat'];
load(fName);

% Setup plot axes
x = linspace(-180,180,nChans+1);
nTimes = size(tctf,1); nBins = length(x);
X = repmat(x',1,nTimes);
Y = repmat(times,nBins,1);

% plot settings
ts  =15;
view3d = [90 90]; % view angle of 3d plots
lim = [-.1 0.8];

% Plot time-resolved total tuning function
FigHandle = figure('Position', [100, 100, 500, 160]); % size of the plot
surf(X,Y,tctf','EdgeColor','none','LineStyle','none','FaceLighting','phong')
shading interp
h=findobj('type','patch');
set(h,'linewidth',2)
hold on
set(gca, 'box','off')
set(gca,'color','none')
set(gca,'LineWidth',2,'TickDir','out');
set(gca,'FontSize',fs)
view(view3d)
axis([x(1) x(end) times(1) times(end) lim]);
set(gca,'XTick',[-180:90:180])
set(gca,'YTick',[-500:500:2000])
xlabel({'Channel Offset'}); set(get(gca,'xlabel'),'FontSize',labelSize);
ylabel('Time (ms)'); set(get(gca,'ylabel'),'FontSize',labelSize);
set(get(gca,'xlabel'),'rotation',90); %where angle is in degrees
c = colorbar;
ax = gca;
axpos = ax.Position;
cpos = c.Position;
cpos(3) = 0.3*cpos(3);
c.Position = cpos;
ax.Position = axpos;
ylabel(c, 'Channel Response'); 
set(c,'fontsize',fs);

% Plot average CTF with bootstrapped SE
FigHandle = figure('Position', [100, 100, 220, 200]); % size of the plot
shadedPlot(x,mCTF-boot.SE,mCTF+boot.SE,[.6 .6 .6],[.6 .6 .6])
hold on;
plot(x,mCTF,'k','LineWidth',2)
set(gca,'XTick',[-180:90:180])
set(gca, 'box','off')
set(gca,'color','none')
set(gca,'LineWidth',1,'TickDir','out');
set(gca,'FontSize',fs)
xlabel({'Channel Offset'}); set(get(gca,'xlabel'),'FontSize',labelSize);
ylabel('Channel Response'); set(get(gca,'ylabel'),'FontSize',labelSize);