% Fig 2d in Foster et al. Psych Science

set(0,'DefaultAxesFontName', 'HelveticaNeue LT 57 Cn')
fs = 9; % size for axis numbers
labelSize = 10; % size for axis labels

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

fName = [dRoot,'bootstrappedCTFslope.mat'];
load(fName); % load data!

% plot properties
times = -500:4:1248;

% load significance marker
fName = [dRoot,'PermTest_Alpha.mat'];
load(fName);
sig = .16*tl.sig;
sig(sig == 0) = nan;

% Plot average CTF with bootstrapped SE
FigHandle = figure('Position', [100, 100, 380, 180]); % size of the plot
shadedPlot(times,ctfSlope.mn-ctfSlope.boot.SE,ctfSlope.mn+ctfSlope.boot.SE,[.6 .6 .6],[.6 .6 .6])
hold on;
plot(times,ctfSlope.mn,'k','LineWidth',2)
scatter(times,sig,1,'r')
ylim([-0.05 0.16])
xlim([-500 1250])
set(gca,'XTick',[-500:500:1250])
set(gca, 'box','off')
set(gca,'color','none')
set(gca,'LineWidth',1,'TickDir','out');
set(gca,'FontSize',fs)
set(gca,'FontName','Arial Narrow')
xlabel({'Times (ms)'}); set(get(gca,'xlabel'),'FontSize',labelSize);
ylabel('Alpha CTF Slope'); set(get(gca,'ylabel'),'FontSize',labelSize);
