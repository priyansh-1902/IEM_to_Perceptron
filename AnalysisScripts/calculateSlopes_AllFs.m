function calculateSlopes_AllFs(sn)
%
% Calculate CTF slopes for real and permuted data
%
% Checked by Dave Sutterer (via dramatic reading), 8.31.2015
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% August 17, 2015

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];


% grab subject's data
fName = [dRoot,num2str(sn), '_SpatialEM_AllFs_Permed.mat'];
load(fName)
rDat.evoked = squeeze(mean(mean(em.tfs.evoked,2),4)); % average across iteration and cross-validation blocks
rDat.total = squeeze(mean(mean(em.tfs.total,2),4));
pDat.evoked = squeeze(mean(em.permtfs.evoked,2)); % average across iterations
pDat.total = squeeze(mean(em.permtfs.total,2));

% Specify properties
nChans = em.nChans; % # of location channels
nPerms = 1000; % % of permutations
nSamps = em.dSamps; % # of samples (after downsampling)
nFreqs = em.nFreqs; % # of frequencies

% preallocate arrays for slope values
rSl.evoked = nan(nFreqs,nSamps); rSl.total = rSl.evoked;
pSl.evoked = nan(nFreqs,nSamps,nPerms); pSl.total = pSl.evoked;

% real evoked data
for f = 1:nFreqs
    for samp = 1:nSamps;
        dat = squeeze(rDat.evoked(f,samp,:));
        x = 1:5;
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        rSl.evoked(f,samp)= fit(1);
    end
end

% real total data
for f = 1:nFreqs
    for samp = 1:nSamps;
        dat = squeeze(rDat.total(f,samp,:));
        x = 1:5;
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        rSl.total(f,samp)= fit(1) ;
    end
end

% permuted evoked data
for perm = 1:nPerms
    for f = 1:nFreqs
        for samp = 1:nSamps;
            dat = squeeze(pDat.evoked(f,perm,samp,:));
            x = 1:5;
            d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
            fit = polyfit(x,d,1);
            pSl.evoked(f,samp,perm)= fit(1) ;
        end
    end
end

% permuted total data
for perm = 1:nPerms
    for f = 1:nFreqs
        for samp = 1:nSamps;
            dat = squeeze(pDat.total(f,perm,samp,:));
            x = 1:5;
            d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
            fit = polyfit(x,d,1);
            pSl.total(f,samp,perm)= fit(1) ;
        end
    end
end

% save slope matrices
filename = [dRoot,num2str(sn),'CTFslopes_AllFs.mat'];
save(filename,'rSl','pSl','-v7.3');

