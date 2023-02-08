% bootstrap SEs for alpha CTF slope for figure 1.
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% July 10, 2016

% specify subjects to plot
load('SubList.mat');
nSubs = length(subs);

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

% settings 
bIter = 10000; % # of bootstrap iterations
nSamps = 438; % # of sample points

root = pwd;

% loop through participants and grab data
for i = 1:nSubs
   
    sn = subs(i);
    
    % load CTF data
    fName = [dRoot,num2str(sn),'_SpatialTF.mat']; load(fName);
    fois = ismember(em.bands,'Alpha');
    t_ctf(i,:,:) = squeeze(mean(mean(em.tfs.total(fois,:,:,:,:),4),2));  % same for total data.   
end

%-------------------------------------------------------------------------%
% Alpha CTF Slope
%-------------------------------------------------------------------------%

% preallocation slope matrix
slopes = nan(nSubs,nSamps);

% calculate slope values for each subject across time
for sub = 1:nSubs
    for samp = 1:nSamps
        dat = squeeze(t_ctf(sub,samp,:));
        x = 1:5;
        d = [dat(1),mean([dat(2),dat(8)]),mean([dat(3),dat(7)]),mean([dat(4),dat(6)]),dat(5)];
        fit = polyfit(x,d,1);
        slopes(sub,samp)= fit(1);
    end
end

boot.IDX = nan(bIter,nSubs);
boot.SLOPE = nan(bIter,nSubs,nSamps);
boot.M = nan(bIter,nSamps);

% loop through bootstrap replications
for b = 1:bIter
    fprintf('Bootstrap replication %d out of %d\n', b, bIter)
    
    [bSLOPE idx] = datasample(slopes,nSubs,1); % sample nSubs many observations from realSl for the subs dimensions (with replacement)
    boot.IDX(b,:) = idx;      % save bootstrap sample index
    boot.SLOPE(b,:,:) = bSLOPE;   % save bootstrapped CTFs
    boot.M(b,:) = mean(bSLOPE); % get the mean osbserved slope (across subs)
    
end

% calculate the bootstrapped SE
boot.SE = std(boot.M);

% calculate actual mean CTF
ctfSlope.mn = mean(slopes);

% save slopes matrix
ctfSlope.slopes = slopes;

% save bootstrap variables
ctfSlope.boot = boot;


% save data
fName = [dRoot,'bootstrappedCTFslope.mat'];
save(fName,'ctfSlope','-v7.3');






