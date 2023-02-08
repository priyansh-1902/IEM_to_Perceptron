function compileBehavior
% Compile behavior file.
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
bRoot = [root(1:end-length(out)),'Behavior/'];

for s = 1:nSubs
    % Loop over subjects
    sn = subs(s);
    fprintf('Subject:\t%d\n',sn)
    fName = [dRoot,num2str(subs(s)),'_Behavior.mat'];
    
    % Vectors to store accuracy data
    acc.allTrials = []; acc.valid = []; acc.invalid = [];
    
    % Indexing vectors
    ind.cueBin = []; ind.targBin = []; ind.cueValidity = []; ind.chanDist = []; ind.chanOffset = []; ind.digit = [];
    
    fn = dir([bRoot,num2str(sn),'SpatialAttn_Exp1_*.mat']);    % grab the files for each subject (*** not to be confused: this title IS correct, forgot to change).
    [runs,c]=size(fn);
    
    % Loop over runs
    for r=1:runs
        fprintf('Crunching Run:\t%d\n', r)
        load([bRoot,num2str(sn),'SpatialAttn_Exp1_',num2str(r),'.mat']);    % read in the data      
        
        % accuracy variables
        acc.allTrials = [acc.allTrials stim.acc];
        acc.valid = [acc.valid stim.acc(stim.cueValidity == 1)];
        acc.invalid = [acc.invalid stim.acc(stim.cueValidity == 0)];
 
        % indexing variables
        ind.cueBin = [ind.cueBin stim.cueChan];
        ind.targBin = [ind.targBin stim.targChan];
        ind.cueValidity = [ind.cueValidity stim.cueValidity];
        ind.chanDist = [ind.chanDist stim.invalidChanDist];
        ind.chanOffset = [ind.chanOffset stim.invalidChanOffset];
        ind.digit = [ind.digit stim.targInd];  
    end
    
    % save data
    save(fName,'ind','acc');      
    
end
