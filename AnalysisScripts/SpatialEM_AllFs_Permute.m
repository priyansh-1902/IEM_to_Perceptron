function SpatialEM_AllFs_Permute(sn,nCores)
%
% Run spatial encoding model across all frequency bands with permuted
% position labels within each block. em.nPerms sets the number of
% permutations.
%
% Inputs
% sn: subject number
% acrpolis: 0 = run on local machine, 1 = run on acropolis
%
% Cheched by Dave Sutter 8.13.2015
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% August 12, 2015

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];
eRoot = [root(1:end-length(out)),'EEG/'];
bRoot = [root(1:end-length(out)),'Behavior/'];

% open matlabpool on acropolis
pc = parcluster('local')
dir = ['/home/joshuafoster/SpatialAttn_TimeFreq/Exp1a/MatlabFiles/SpatialEM_AllFs_Permeute_Sub',num2str(sn)];
mkdir(dir);
pc.JobStorageLocation = dir;
matlabpool(pc,nCores);

% determine random seed
rng shuffle; % get new random seed
em.randSeed = rng; % save random seed to em structure

name = '_SpatialEM_AllFs_Permed.mat'; % name of files to be saved

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];
eRoot = [root(1:end-length(out)),'EEG/'];
bRoot = [root(1:end-length(out)),'Behavior/'];

% parameters to set
em.nPerms = 1000; % # of permutations
nPerms = em.nPerms;


% Grab data------------------------------------------------------------

% Grab EM data file
fName = [dRoot, num2str(sn),'_SpatialEM_AllFs.mat'];
load(fName)

% get analysis settings from data file
nChans = em.nChans;
nBins = em.nBins;
nIter = em.nIter;
nBlocks = em.nBlocks;
freqs = em.freqs;
dtimes = em.dtime;
nFreqs = em.nFreqs;
nElectrodes = em.nElectrodes;
nSamps = em.nSamps;
dSamps = em.dSamps;stepSize = em.stepSize;
Fs = em.Fs;
window = em.window;
nTrialsPerBlock = em.nTrialsPerBlock;
basisSet = em.basisSet;
posBin = em.posBin;

% Get EEG data
fName = [eRoot, num2str(sn), '_EEG.mat']; load(fName);
eegs = eeg.data(:,1:20,:); % get scalp EEG (drop EOG electrodes)
artInd = eeg.arf.artIndCleaned.'; % grab artifact rejection index
tois = ismember(eeg.preTime:4:eeg.postTime,em.time); nTimes = length(tois); % index time points for analysis.

% Remove rejected trials
eegs = eegs(~artInd,:,:);
posBin = posBin(~artInd);

em.nTrials = length(posBin); nTrials = em.nTrials; % # of good trials
%----------------------------------------------------------------------

% Preallocate Matrices
tf_evoked = nan(nFreqs,nIter,nPerms,dSamps,nChans); tf_total = tf_evoked;
% tf_evoked = nan(nFreqs,nIter,nPerms,dSamps,nBlocks,nChans); tf_total = tf_evoked;
permInd = nan(nFreqs,nIter,nPerms,nBlocks,nTrialsPerBlock);

% load in block assignment
blocks = em.blocks;


% Loop through each frequency
for f = 1:nFreqs
    
    fprintf('Frequency %d out of %d\n',f ,nFreqs);
    tic
    
    % Filter Data
    fdata_evoked = nan(nTrials,nElectrodes,nTimes);
    fdata_total = nan(nTrials,nElectrodes,nTimes);
    parfor c = 1:nElectrodes
        fdata_evoked(:,c,:) = hilbert(eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(f,1),freqs(f,2))')';
        fdata_total(:,c,:) = abs(hilbert(eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(f,1),freqs(f,2))')').^2; % instantaneous power calculated here for induced activity.
    end
    
    % trim filtered data to remove times that are not of interest (after filtering to avoid edge artifacts)
    fdata_evoked = fdata_evoked(:,:,tois);
    fdata_total = fdata_total(:,:,tois);
    
    % downsample to reduced sampled rate (after filtering, so that downsampling doesn't affect filtering)
    fdata_evoked = fdata_evoked(:,:,1:stepSize:nSamps);
    fdata_total = fdata_total(:,:,1:stepSize:nSamps);
    
    % Loop through each iteration
    for iter = 1:nIter
        
        % grab block assigment for current iteration
        blocks = em.blocks(:,iter);
        
        % Loop through permutations
        for perm = 1:nPerms

            %-----------------------------------------------------------------------------
            % Permute trial assignment within each block
            %-----------------------------------------------------------------------------
            permedPosBin = nan(size(posBin)); % preallocate permuted position bins vector
            for b = 1:nBlocks % for each block..
                pInd = randperm(nTrialsPerBlock); % create a permutation index
                permedBins(pInd) = posBin(blocks == b); % grab block b data and permute according data according to the index
                permedPosBin(blocks == b) = permedBins; % put permuted data into permedPosBin
                permInd(f,iter,perm,b,:) = pInd; % save the permutation (permInd is saved at end of the script)
            end
            %-----------------------------------------------------------------------------
            
            % Average data for each position bin across blocks
            posBins = 1:nBins;
            blockDat_evoked = nan(nBins*nBlocks,nElectrodes,dSamps); % averaged evoked data
            blockDat_total = nan(nBins*nBlocks,nElectrodes,dSamps);  % averaged total data
            labels = nan(nBins*nBlocks,1);                           % bin labels for averaged data
            blockNum = nan(nBins*nBlocks,1);                         % block numbers for averaged data
            c = nan(nBins*nBlocks,nChans);                           % predicted channel responses for averaged data
            bCnt = 1;
            for ii = 1:nBins
                for iii = 1:nBlocks
                    blockDat_evoked(bCnt,:,:) = abs(squeeze(mean(fdata_evoked(permedPosBin==posBins(ii) & blocks==iii,:,:),1))).^2;
                    blockDat_total(bCnt,:,:) = squeeze(mean(fdata_total(permedPosBin==posBins(ii) & blocks==iii,:,:),1));
                    labels(bCnt) = ii;
                    blockNum(bCnt) = iii;
                    c(bCnt,:) = basisSet(ii,:);
                    bCnt = bCnt+1;
                end
            end
            
            for t = 1:dSamps
                
                % grab data for timepoint t
                toi = ismember(dtimes,dtimes(t)-window/2:dtimes(t)+window/2); % time window of interest
                de = squeeze(mean(blockDat_evoked(:,:,toi),3)); % evoked data
                dt = squeeze(mean(blockDat_total(:,:,toi),3));  % total data
                
                % Do forward model
                tmpeCR = nan(nBlocks,nChans); tmptCR = nan(nBlocks,nChans); % for shifted channel respones
                
                for i=1:nBlocks % loop through blocks, holding each out as the test set
                    
                    trnl = labels(blockNum~=i); % training labels
                    tstl = labels(blockNum==i); % test labels
                    
                    %-----------------------------------------------------%
                    % Analysis on Evoked Power                            %
                    %-----------------------------------------------------%
                    B1 = de(blockNum~=i,:);    % training data
                    B2 = de(blockNum==i,:);    % test data
                    C1 = c(blockNum~=i,:);     % predicted channel outputs for training data
                    W = C1\B1;                 % estimate weight matrix
                    C2 = (W'\B2')';            % estimate channel responses
                    
                    % shift eegs to common center
                    n2shift = ceil(size(C2,2)/2);
                    for ii=1:size(C2,1)
                        [~, shiftInd] = min(abs(posBins-tstl(ii)));
                        C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
                    end
                     
                    tmpeCR(i,:) = mean(C2);
%                     tf_evoked(f,iter,perm,t,i,:) = mean(C2,1); % average shifted channel responses
                    
                    %-----------------------------------------------------%
                    % Analysis on Total Power                             %
                    %-----------------------------------------------------%
                    B1 = dt(blockNum~=i,:);    % training data
                    B2 = dt(blockNum==i,:);    % test data
                    C1 = c(blockNum~=i,:);     % predicted channel outputs for training data
                    W = C1\B1;                 % estimate weight matrix
                    C2 = (W'\B2')';            % estimate channel responses
                    
                    % shift eegs to common center
                    n2shift = ceil(size(C2,2)/2);
                    for ii=1:size(C2,1)
                        [~, shiftInd] = min(abs(posBins-tstl(ii)));
                        C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
                    end
                    
                    tmptCR(i,:) = mean(C2);
%                     tf_total(f,iter,perm,t,i,:) = mean(C2,1); % average shifted channel responses
                    
                    %-----------------------------------------------------%
                end
                tf_evoked(f,iter,perm,t,:) = mean(tmpeCR);
                tf_total(f,iter,perm,t,:) = mean(tmptCR);
            end
        end
    end
    toc % stop timing the frequency loop
end

fName = [dRoot,num2str(sn),name];
em.permInd = permInd;
em.permtfs.evoked = tf_evoked;
em.permtfs.total = tf_total;
save(fName,'em','-v7.3');

matlabpool close
