function calculateSlopes_Alpha(sn)
%
% Calculate CTF slopes for real and permuted data
%
% Not yet checked by Dave
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% August 24, 2015

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];


% grab subject's data
fName = [dRoot,num2str(sn), '_SpatialTF_Permed.mat'];
load(fName)
rDat.evoked = squeeze(mean(mean(em.tfs.evoked,2),4)); % average across iteration and cross-validation blocks
rDat.total = squeeze(mean(mean(em.tfs.total,2),4));
pDat.evoked = squeeze(mean(em.permtfs.evoked,2)); % average across iterations
pDat.total = squeeze(mean(em.permtfs.total,2));

% Specify properties
nChans = em.nChans; % # of location channels
nPerms = 1000; % % of permutations
nSamps = length(em.time); % # of samples (after downsampling)
nFreqs = 1; % # of frequencies

% preallocate arrays for slope values
rSl.evoked = nan(nSamps); rSl.total = rSl.evoked;
pSl.evoked = nan(nSamps,nPerms); pSl.total = pSl.evoked;

% real evoked data
for samp = 1:nSamps;
    dat = squeeze(rDat.evoked(samp,:));
    x = 1:5;
    d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
    fit = polyfit(x,d,1);
    rSl.evoked(samp)= fit(1) ;
end

% real total data
for samp = 1:nSamps;
    dat = squeeze(rDat.total(samp,:));
    x = 1:5;
    d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
    fit = polyfit(x,d,1);
    rSl.total(samp)= fit(1) ;
end

% permuted evoked data
for perm = 1:nPerms
    for samp = 1:nSamps;
        dat = squeeze(pDat.evoked(perm,samp,:));
        x = 1:5;
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        pSl.evoked(samp,perm)= fit(1) ;
    end
end

% permuted total data
for perm = 1:nPerms
    for samp = 1:nSamps;
        dat = squeeze(pDat.total(perm,samp,:));
        x = 1:5;
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        pSl.total(samp,perm)= fit(1) ;
    end
end

% save slope matrices
filename = [dRoot,num2str(sn),'CTFslopes_Alpha.mat'];
save(filename,'rSl','pSl','-v7.3');

