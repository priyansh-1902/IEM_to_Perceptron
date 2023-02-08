  function SpatialEM_AllFs(sn,nCores)
%
% Run spatial encoding model across all frequency bands (see em.freqs).
%
% The model is not run at every samplev point. Instead, the sample rate for
% the analysis is specified by 'em.sampRate'.
%
% modified 5.6.2016 by J. Foster:
% added new approach for opening matlabpool on acropolis.
% determine random seed for each subject.
%
% Cheked by Dave Sutterer 8.13.2015
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
dir = ['/home/joshuafoster/SpatialAttn_TimeFreq/Exp1a/MatlabFiles/SpatialEM_AllFs_Sub',num2str(sn)];
mkdir(dir);
pc.JobStorageLocation = dir;
matlabpool(pc,nCores);

% determine random seed
rng shuffle; % get new random seed
em.randSeed = rng; % save random seed to em structure

name = '_SpatialEM_AllFs.mat'; % name of files to be saved

% parameters to set
em.nChans = 8; % # of channels
em.nBins = 8; % # of position bins
em.nIter = 5; % # of iterations
em.nBlocks = 3; % # of blocks for cross-validation
fqs = 4:50;     % range of frequency bands
for fbnd = 1:length(fqs)
        em.freqs(fbnd,:) = [fqs(fbnd) fqs(fbnd)+1]; % frequency bands to analyze
end
em.window = 4;
em.Fs = 250;
em.nFreqs = size(em.freqs,1);
em.nElectrodes = 20;
em.sampRate = 20; % downsampled sample rate (in ms)
em.time = -500:4:1248; % specificy time window of interest
em.dtime = -500:em.sampRate:1248; % downsampled time points
em.stepSize = em.sampRate/(1000/em.Fs); % number of samples the classifer jumps with each shift
em.nSamps = length(em.time); %  # of samples
em.dSamps = length(em.dtime); % # of samples at downsampled rate

% for brevity in analysis
nChans = em.nChans;
nBins = em.nBins;
nIter = em.nIter;
nBlocks = em.nBlocks;
freqs = em.freqs;
dtimes = em.dtime;
nFreqs = size(em.freqs,1);
nElectrodes = em.nElectrodes;
nSamps = em.nSamps;
dSamps = em.dSamps;
stepSize = em.stepSize;
Fs = em.Fs;
window = em.window;

% Specify basis set
em.sinPower = 7;
em.x = linspace(0, 2*pi-2*pi/nBins, nBins);
em.cCenters = linspace(0, 2*pi-2*pi/nChans, nChans);
em.cCenters = rad2deg(em.cCenters);
pred = sin(0.5*em.x).^em.sinPower; % hypothetical channel responses
pred = wshift('1D',pred,5); % shift the initial basis function
basisSet = nan(nChans,nBins);
for c = 1:nChans;
    basisSet(c,:) = wshift('1D',pred,-c); % generate circularly shifted basis functions
end
em.basisSet = basisSet; % save basis set to data structure


% Grab data------------------------------------------------------------

% Get position bin index from behavior file
fName = [dRoot, num2str(sn), '_Behavior.mat']; load(fName);
em.posBin = ind.cueBin'; % add to class structure so it's saved
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
tf_evoked = nan(nFreqs,nIter,dSamps,nBlocks,nChans); tf_total = tf_evoked;
em.blocks = nan(nTrials,nIter);  % create em.block to save block assignments

%--------------------------------------------------------------------------
% Create block assignment for each iteration
%--------------------------------------------------------------------------
% trials are assigned to blocks so that # of trials per position are equated within blocks
% this is done before the frequency loop so that the same blocks assignments are used for all freqs 

for iter = 1:nIter
    
    % preallocate arrays
    blocks = nan(size(posBin));
    shuffBlocks = nan(size(posBin));
    
    % count number of trials within each position bin
    clear binCnt
    for bin = 1:nBins
        binCnt(bin) = sum(posBin == bin);
    end
    
    minCnt = min(binCnt); % # of trials for position bin with fewest trials
    nPerBin = floor(minCnt/nBlocks); % max # of trials such that the # of trials for each bin can be equated within each block
    
    % shuffle trials
    shuffInd = randperm(nTrials)'; % create shuffle index
    shuffBin = posBin(shuffInd); % shuffle trial order
    
    % take the 1st nPerBin x nBlocks trials for each position bin.
    for bin = 1:nBins;
        idx = find(shuffBin == bin); % get index for trials belonging to the current bin
        idx = idx(1:nPerBin*nBlocks); % drop excess trials
        x = repmat(1:nBlocks',nPerBin,1); shuffBlocks(idx) = x; % assign randomly order trials to blocks
    end
    
    % unshuffle block assignment
    blocks(shuffInd) = shuffBlocks;
    
    % save block assignment
    em.blocks(:,iter) = blocks; % block assignment
    em.nTrialsPerBlock = length(blocks(blocks == 1)); % # of trials per block
    
end
%--------------------------------------------------------------------------

% Loop through each frequency
for f = 1:nFreqs
   
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
                blockDat_evoked(bCnt,:,:) = abs(squeeze(mean(fdata_evoked(posBin==posBins(ii) & blocks==iii,:,:),1))).^2;
                blockDat_total(bCnt,:,:) = squeeze(mean(fdata_total(posBin==posBins(ii) & blocks==iii,:,:),1));
                labels(bCnt) = ii;
                blockNum(bCnt) = iii;
                c(bCnt,:) = basisSet(ii,:);
                bCnt = bCnt+1;
            end
        end
        
        parfor t = 1:dSamps
            
            % grab data for timepoint t
            toi = ismember(dtimes,dtimes(t)-window/2:dtimes(t)+window/2); % time window of interest
            de = squeeze(mean(blockDat_evoked(:,:,toi),3)); % evoked data
            dt = squeeze(mean(blockDat_total(:,:,toi),3));  % total data
            
            % Do forward model
            
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
                
                tf_evoked(f,iter,t,i,:) = mean(C2,1); % average shifted channel responses
                
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
                
                tf_total(f,iter,t,i,:) = mean(C2,1); % average shifted channel responses
                
                %-----------------------------------------------------%
            end
        end
    end
end

fName = [dRoot,num2str(sn),name];
em.tfs.evoked = tf_evoked;
em.tfs.total = tf_total;
save(fName,'em','-v7.3');

matlabpool close