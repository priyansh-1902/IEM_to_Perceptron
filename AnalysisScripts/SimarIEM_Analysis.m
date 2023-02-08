%%% IEM _ Using Online EEG Data:

% Lets try and look at one participant at a time...:
root = pwd;out = 'AnalysisScripts';
load([root,'\EEG\1_EEG.mat'])
load([root,'\data\1_SpatialTF.mat'])
load([root,'\data\1_Behavior.mat'])
load([root,'\data\test_c.mat'])


eegs = eeg.data(:,1:20,:); % get scalp EEG (drop EOG electrodes)
artInd = eeg.arf.artIndCleaned.'; % grab artifact rejection index
tois = ismember(eeg.preTime:4:eeg.postTime,em.time); nTimes = length(tois); % index time points for analysis.
% em.time = -500:4:1248; % time points of interest

% Remove rejected trials
eegs = eegs(~artInd,:,:);
posBin = em.posBin(~artInd); % This gives me the position for the item for the particular trial... which I need to index the BasisChannel for each trial to get nTrial x Channel
nTrials = length(posBin); % # of good trials

% Channel Response Matrix (n stimuli (row) x k channels (col)) * 
% Weight Matrix (k Channels x e Electrodes) =
% EEG response (n stimuli (trial) x e Electrode)

% EEG Responses can be found under variable eegs where the first dimension
% is trials, then electrodes, then timepoints...
% I want to get the Channel Response Matrix:

orientations = [0:7];
figure;
plot(orientations,em.basisSet);

% compute the channelResponse for each trial
for iTrial = 1:nTrials
  channelResponse(iTrial,:) = em.basisSet(posBin(iTrial),:);
end

% compute estimated weights; perhaps I need to select specific time windows
% and average across them, because now I have the Channel responses for
% participant 1 across all the trials, and I need to now solve for weights
% given the eegResponse:

eegResponse = eegs(:,:,1);

estimatedWeights =  pinv(channelResponse) * eegResponse;

% We now have a estimated weight!!!


% In real life, what we want to do is split the trials, randomly in half,
% to assign half the trials as training data and the other half as testing
% data. And we want to do this for every time point. 
