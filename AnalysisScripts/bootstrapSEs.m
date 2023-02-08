function bootstrapSEs
%
% Bootstrapping procedure:
%
% 1. Select bootstrap sample:
%    Draw bIter many independent bootstrap samples, consisting of nSubs many
%    subjects sampled with replacement.
%
% 2. Evaluate bootstrap replication:
%    For each bootstrap sample, calculate the channel responses.
%
% 3. Estimate the standar error by the sample standard deviation of the
%    bootstrap replications:
%    Calculate the sd of the channel responses across the bootstrapped replications
%
%    Simple as that!!
%
% For more information on bootstrapping procedure see page 47 of:
% Efron, B. & Tibshirani, R.J. An Introduction to the Bootstrap. Monographs
% on Statistics and Probability 57. (1993).
%
% Last modifed by J. Foster 7.9.2016: removed plotting lines and created a
% separate plotting function.
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% August 20, 2015

% specify subjects to plot
subs = [1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20];
nSubs = length(subs);

% specify filename of data to grab
name = '_SpatialTF.mat';

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];

bIter = 10000; % # of bootstrap replications (the more the better)
nChans = 8;

% loop through participants and grab tf generated by forward model
for i = 1:nSubs
   
    sn = subs(i);
    fName = [dRoot,num2str(sn),name];
    load(fName)
    fois = ismember(em.bands,'Alpha');
    t_ctf(i,:,:) = squeeze(mean(mean(em.tfs.total(fois,:,:,:,:),4),2));  % same for total data.

end

% average across the period after stim onset
ctf = squeeze(mean(t_ctf(:,201:438,:),2));

% preallocate matrices
boot.IDX = nan(bIter,nSubs);
boot.CTF = nan(bIter,nSubs,nChans);
boot.M = nan(bIter,nChans)

    % loop through bootstrap replications
    for b = 1:bIter
        fprintf('Bootstrap replication %d out of %d\n', b, bIter)       
        
        [bCTF idx] = datasample(ctf,nSubs,1); % sample nSubs many observations from realSl for the subs dimensions (with replacement)
        boot.IDX(b,:) = idx;      % save bootstrap sample index
        boot.CTF(b,:,:) = bCTF;   % save bootstrapped CTFs
        boot.M(b,:) = mean(bCTF); % get the mean osbserved slope (across subs)   
     
    end
    
    % calculate the bootstrapped SE
    boot.SE = std(boot.M); boot.SE = [boot.SE boot.SE(1)]; % mirroring chan 1 on other side
    
    % calculate actual mean CTF
    mCTF = mean(ctf); mCTF = [mCTF, mCTF(1)];
    
    % time-resolved mean CTF
    tctf = squeeze(mean(t_ctf,1)); tctf = [tctf tctf(:,1)];
    
    % save data
    fName = [dRoot,'bootstrappedCTFs.mat'];
    save(fName,'tctf','mCTF','boot','-v7.3');

    