root = pwd; out = "Priyansh's Scripts";

Fs = 250;
freqs = [8 12];
nFreqs = size(freqs,1);

for i = 1:20
    %eeg.total = eeg.data;
    dataFile = [num2str(i) '_EEG.mat']; %#ok<ST2NM> 
    load(dataFile)
    eegs = eeg.data(:,1:20,:); % get scalp EEG (drop EOG electrodes)
    artInd = eeg.arf.artIndCleaned.'; % grab artifact rejection index
    eegs = eegs(~artInd,:,:);
    
    s = size(eegs);
    s1 = s(1);
    s2 = s(2);
    s3 = s(3);

    eeg.evoked = nan(s1, s2, s3);
    
    for f = 1:nFreqs
    
        fprintf('Frequency %d out of %d\n', f, nFreqs)

        for electrode = 1:s2
            eeg.evoked(:, electrode, :) =  hilbert(eegfilt(squeeze(eegs(:,electrode,:)),Fs,freqs(f,1),freqs(f,2))')';
        end
    end
    
    save([num2str(i) '_EEGfilt'])
    fprintf('saved\n')
        
end

