function SpatialCueingEffect
% Calculate spatial cueing effect in accuracy (excluding trials with EEG artifacts).
%
% Joshua J. Foster
% joshua.james.foster@gmail.com
% University of Chicago
% April 20, 2016

subs = [1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20];
nSubs = length(subs);

% setup directories
root = pwd; out = 'AnalysisScripts';
dRoot = [root(1:end-length(out)),'Data/'];
eRoot = [root(1:end-length(out)),'EEG/'];
bRoot = [root(1:end-length(out)),'Behavior/'];

name = [dRoot,'SpatialCueingEffect.mat']; % name of files to be saved

% preallocate matrix
accuracy = nan(nSubs,2);
nTrials = nan(nSubs,1);

for s = 1:nSubs
    
    sn = subs(s)
    
    % Get position bin index from behavior file
    fName = [dRoot, num2str(sn), '_Behavior.mat']; load(fName);
    validity = ind.cueValidity;
    acc = acc.allTrials;
   
    % Get EEG data
    fName = [eRoot, num2str(sn), '_EEG.mat']; load(fName);
    artInd = eeg.arf.artIndCleaned.'; % grab artifact rejection index
    
    % Remove rejected trials
    validity = validity(~artInd);
    acc = acc(~artInd);
    nTrials(s) = length(acc);
    
    % calculate accuracy
    valid = sum(acc(validity == 1))/length(acc(validity == 1));
    invalid = sum(acc(~validity))/length(acc(~validity));
    accuracy(s,1) = valid;
    accuracy(s,2) = invalid;
    
   
end

% save number of trials included in analysis (should match that from EEG
% data)

save(name,'accuracy','subs','nTrials','-v7.3');
    