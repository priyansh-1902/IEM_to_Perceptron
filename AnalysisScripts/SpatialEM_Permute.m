function SpatialEM_Permute(sn,nCores)
%
% Run spatial encoding model on evoked and total alpha power with permuted
% trials labels within each block (# of permutations set by em.nPerms)
%
% Inspected by David Sutterer on 6.28.2015.
%
% modifed 4.29.2016 by J. Foster:
% 1. generate and save random seed (lines 17-19)
% 2. remove subject loop (run in a wrapper function
%
% modifed 5.3.2016 by J. Foster after running on acropolis:
% just added matlabpool close at end of script
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Oregon
% June 26, 2015

matlabpool('local',nCores)

% determine random seed
rng shuffle; % get new random seed
em.randSeed = rng; % save random seed to em structure

em.nPerms = 1000;
nPerms = em.nPerms;

name = '_SpatialTF_Permed.mat'; % name of files to be saved

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];
eRoot = [root(1:end-length(out)),'EEG/'];
bRoot = [root(1:end-length(out)),'Behavior/'];



% Grab TF data file
fName = [dRoot, num2str(sn), '_SpatialTF.mat'];
load(fName);

% get analysis settings from TF data file.
nChans = em.nChans;
nBins = em.nBins;
nIter = em.nIter;
nBlocks = em.nBlocks;
freqs = em.frequencies;
times = em.time;
nFreqs = size(em.frequencies,1);
nElectrodes = em.nElectrodes;
nSamps = length(em.time);
Fs = em.Fs;
basisSet = em.basisSet;
posBin = em.posBin;
nTrialsPerBlock = em.nTrialsPerBlock;

% Grab data------------------------------------------------------------

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
tf_evoked = nan(nFreqs,nIter,nPerms,nSamps,nChans); tf_total = tf_evoked;
C2_evoked = nan(nFreqs,nIter,nPerms,nSamps,nBins,nChans); C2_total = C2_evoked;
permInd = nan(nFreqs,nIter,nPerms,nBlocks,nTrialsPerBlock);
permedBins = nan(1,nTrialsPerBlock);

% Loop through each frequency
for f = 1:nFreqs
    fprintf('Frequency %d out of %d\n', f, nFreqs)
    
    % Filter Data
    fdata_evoked = nan(nTrials,nElectrodes,nTimes);
    fdata_total = nan(nTrials,nElectrodes,nTimes);
    parfor c = 1:nElectrodes
        fdata_evoked(:,c,:) = hilbert(eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(f,1),freqs(f,2))')';
        fdata_total(:,c,:) = abs(hilbert(eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(f,1),freqs(f,2))')').^2; % instantaneous power calculated here for induced activity.
    end
    
    % Loop through each iteration
    for iter = 1:nIter
        
        blocks = em.blocks(:,iter); % grab blocks assignment for current iteration
        
        % Loop through permutations
        for perm = 1:nPerms
            tic % start timing permutation loop
            fprintf('Permutation %d out of %d\n',perm,nPerms);
            
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
            blockDat_evoked = nan(nBins*nBlocks,nElectrodes,nSamps); % averaged evoked data
            blockDat_total = nan(nBins*nBlocks,nElectrodes,nSamps);  % averaged total data
            labels = nan(nBins*nBlocks,1);                           % bin labels for averaged data
            blockNum = nan(nBins*nBlocks,1);                         % block numbers for averaged data
            c = nan(nBins*nBlocks,nChans);                           % predicted channel responses for averaged data
            bCnt = 1;
            for ii = 1:nBins
                for iii = 1:nBlocks
                    blockDat_evoked(bCnt,:,:) = abs(squeeze(mean(fdata_evoked(permedPosBin==posBins(ii) & blocks==iii,:,tois),1))).^2;
                    blockDat_total(bCnt,:,:) = squeeze(mean(fdata_total(permedPosBin==posBins(ii) & blocks==iii,:,tois),1));
                    labels(bCnt) = ii;
                    blockNum(bCnt) = iii;
                    c(bCnt,:) = basisSet(ii,:);
                    bCnt = bCnt+1;
                end
            end
            
            parfor t = 1:nSamps
                
                % grab data for timepoint t
                toi = ismember(times,times(t)-em.window/2:times(t)+em.window/2); % time window of interest
                de = squeeze(mean(blockDat_evoked(:,:,toi),3)); % evoked data
                dt = squeeze(mean(blockDat_total(:,:,toi),3));  % total data
                
                % Do forward model
                tmpeC2 = nan(nBlocks,nBins,nChans); tmptC2 = tmpeC2; % for unshifted channel responses
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
                    W = C1\B1;          % estimate weight matrix
                    C2 = (W'\B2')';     % estimate channel responses
                    
                    % tmpeC2(i,:,:) = C2;
                    
                    % shift eegs to common center
                    n2shift = ceil(size(C2,2)/2);
                    for ii=1:size(C2,1)
                        [~, shiftInd] = min(abs(posBins-tstl(ii)));
                        C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
                    end
                    
                    tmpeCR(i,:) = mean(C2); % average shifted channel responses
                    
                    %-----------------------------------------------------%
                    % Analysis on Total Power                             %
                    %-----------------------------------------------------%
                    B1 = dt(blockNum~=i,:);    % training data
                    B2 = dt(blockNum==i,:);    % test data
                    C1 = c(blockNum~=i,:);     % predicted channel outputs for training data
                    W = C1\B1;          % estimate weight matrix
                    C2 = (W'\B2')';     % estimate channel responses
                    
                    % tmptC2(i,:,:) = C2;
                    
                    % shift eegs to common center
                    n2shift = ceil(size(C2,2)/2);
                    for ii=1:size(C2,1)
                        [~, shiftInd] = min(abs(posBins-tstl(ii)));
                        C2(ii,:) = wshift('1D', C2(ii,:), shiftInd-n2shift-1);
                    end
                    
                    tmptCR(i,:) = mean(C2); % averaged shifted channel responses
                    
                    %-----------------------------------------------------%
                    
                end
                % save data to indexed matrix
                % C2_evoked(f,iter,perm,t,:,:) = mean(tmpeC2);
                % C2_total(f,iter,perm,t,:,:) = mean(tmptC2);
                tf_evoked(f,iter,perm,t,:) = mean(tmpeCR);
                tf_total(f,iter,perm,t,:) = mean(tmptCR);
            end
            toc
        end
    end
    toc % stop timing the frequency loop
end

fName = [dRoot,num2str(sn),name];
em.permInd = permInd;
% em.permC2.evoked = C2_evoked; not saving these because they're huge!
% em.permC2.total = C2_total;
em.permtfs.evoked = tf_evoked;
em.permtfs.total = tf_total;
save(fName,'em','-v7.3');

matlabpool close

