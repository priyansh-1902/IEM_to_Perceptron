% Fig 2g in Foster et al. Psych Science

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

% load low frequency data
fName = [dRoot,'PermTest_AllFs.mat'];
load(fName);
totaldat =  tl.mn.*tl.sig;

% set figure properties
set(0,'DefaultAxesFontName', 'HelveticaNeue LT 57 Cn')
fs = 9; % size for axis numbers
labelSize = 10; % size for axis labels
ts = 15;
lim = [0 0.15];
t = -500:20:1250;
f = 4:50;

% plot total data
FigHandle = figure('Position', [100, 100, 420, 170]); % size of the plot
imagesc(t,f,totaldat);
hold on
set(gca, 'box','off')
set(gca,'color','none')
set(gca,'LineWidth',1,'TickDir','out');
set(gca,'FontSize',fs)
set(gca,'FontName','Arial Narrow')
%set(gca,'YMinorTick','on')
set(gca,'YDir','normal');
% title('Total Power','FontSize',ts)
xlabel({'Time (ms)'}); set(get(gca,'xlabel'),'FontSize',labelSize);
ylabel('Frequency (Hz)'); set(get(gca,'ylabel'),'FontSize',labelSize);
caxis(lim)
c = colorbar;
ax = gca;
axpos = ax.Position;
cpos = c.Position;
cpos(3) = 0.3*cpos(3);
c.Position = cpos;
ax.Position = axpos;
ylabel(c, 'CTF Slope');
set(c,'fontsize',fs);



