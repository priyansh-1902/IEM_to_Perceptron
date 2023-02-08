function permTest_Alpha
%
% Generate mean slope values for all time x freq pixels. Calculate p-values
% for the one-sample t-stat pm CTF slope for each sample point using a
% surrogate distribution generated with permuted data.
%
% The t-stat is calculated as: t = (m - 0)/SEx
%
% where m is the sample mean slope an SEx is the standard error of the mean
% slope i.e. samp std dev/sqrt(n)
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% August 24, 2015

% specify subjects to plot
subs = [1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20];
nSubs = length(subs);

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

% specify filename of data to plot
name = 'CTFslopes_Alpha.mat';

root = pwd;

% specs
pthresh = .01; % significance threshold
nSamps = 438;
nPerms = 1000;

for signal = 1:2; % 1 = evoked, 2 = total

% load in data
for i = 1:nSubs
    sn = subs(i);
    fName = [dRoot,num2str(sn),name];
    load(fName)
    % load relevant data
    if signal == 1
        realSl(i,:,:) = rSl.evoked;
        surrSl(i,:,:,:) = pSl.evoked;
    else
        realSl(i,:,:) = rSl.total;
        surrSl(i,:,:,:) = pSl.total;
    end
end

% calculate mean slope
mn = squeeze(mean(realSl));

% preallocate matrices
pval = nan(nSamps,1);
sig = zeros(nSamps,1);

% calculate the real one-sample t-stat
realAve = squeeze(mean(realSl));
realSE = squeeze(std(realSl)/sqrt(nSubs));
realT = realAve./realSE;

% calculate the surrograte one-sample t-stats
surrAve = squeeze(mean(surrSl));
surrSE = squeeze(std(surrSl))./sqrt(nSubs);
surrT = surrAve./surrSE;
        
% calculate p-values
for t = 1:nSamps
    surrDist = surrT(t,:);
    pval(t) = length(surrDist(surrDist>realT(t)))/nPerms; % calculate p-value
    if pval(t) < pthresh
        sig(t) = 1; % set sig to one if p-value < pthresh
    end
end

% save relevant variables to structure
if signal == 1
    ev.pthresh = pthresh;
    ev.mn = mn;
    ev.realT = realT;
    ev.surrT = surrT;
    ev.sig = sig;
    ev.pval = pval;
else
    tl.pthresh = pthresh;
    tl.mn = mn;
    tl.realT = realT;
    tl.surrT = surrT;
    tl.sig = sig;
    tl.pval = pval;
end
        
end

fName = [dRoot,'PermTest_Alpha.mat'];
save(fName,'ev','tl','-v7.3');