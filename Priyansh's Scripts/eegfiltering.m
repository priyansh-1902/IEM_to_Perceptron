root = pwd; out = "Priyansh's Scripts";

%eeg.total = eeg.data;
for i = 3
    fprintf([num2str(i) '\n'])
    dataFile = [root '(1)\EEG\' num2str(i) '_EEG.mat']; %#ok<ST2NM> 
    load(dataFile)

    eeg.evoked = eeg.data;

    s = size(eeg.data);
    s1 = s(1);
    s2 = s(2);
    for trial = 1:s1
        
        for electrode = 1:s2
            eeg.evoked(trial, electrode, :) = hilbert(eegfilt(eeg.data(trial, electrode, :), 250, 8, 12));
            %eeg.total(trial, electrode, :) = abs(eeg.evoked(trial, electrode, :));
            
        end
        fprintf([num2str(trial) '\n'])
    end
    
    save(['EEG\' num2str(i) '_EEGfilt'])
    fprintf('saved\n')
    
end


