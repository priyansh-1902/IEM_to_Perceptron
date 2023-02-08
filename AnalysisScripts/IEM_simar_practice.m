% Channel Response Matrix (n stimuli (row) x k channels (col)) * 
% Weight Matrix (k Channels x v Voxels)  
% BOLD response (n stimuli x v Voxels)

% n stimuli length = 8; k channels length = 6; v voxels = 40
%B = C * W;



nStim = 8; kChan = 6; vVox = 40;

C = [];

for i = 1:nStim
    C = [C;randperm(8,kChan)];
end

W = [randperm(50,40);randperm(50,40);randperm(50,40);randperm(50,40);randperm(50,40);randperm(50,40)];

% TRY CHECKING IF YOU NEED TO ROUND...
B = C * W;


W_est =  C\B;
W == W_est
W_est = round(W_est);
W == W_est %? Yes its equal

W_est_2 =  pinv(C) * B; W_est_2 = round(W_est_2);
W == W_est_2; %? Yes it is equal

W_est_3 =  inv(C.'*C)*C.' * B; W_est_3 = round(W_est_3);
W == W_est_3; %? Yes it is equal


% % % % % % % % % % % % % % % % Introduction % % % % % % % % % % % % % % %
% Encoding models can be a powerful way to analyze data because they can
% reduce a high-dimensional stimulus space to a lower dimensional 
% representation based on knowledge of how neural system encode stimuli 
% and thus have more power to extract relevant information from cortical 
% measurements. A channel encoding model, first introduced by 
% Brouwer & Heeger, 2009 for the representation of color has been widely
% used to understand cortical representations of continuous stimulus 
% variables like orientation, direction of motion and space. This tutorial
% will teach you how to simulate data for oriented stimuli that conform 
% to the assumptions of the model, fit and invert the model.
% 
% The basic idea is that each measurement unit (typically a voxel) 
% is modeled to be the linear sum of underlying populations of neurons 
% tuned for different orientations. So, for example if you present some
% orientation and get responses higher (red) or lower (blue) than the mean 
% response, you would assume that higher responses are due to a stronger 
% weighting of neural populations tuned for that orientation, and weaker 
% responses due to stronger weighting of neural populations tuned for 
% other orientations.
% 
% In matrix algebra, what one does is make a channel response matrix
% which contains the channel responses for each of the stimuli in an
% experiment (one row per trial). This gets multiplied by an (unknown) 
% weight matrix that best fits the measured BOLD responses (one row for
% each stimulus). Note that typically you need to get a single amplitude
% for each voxel per trial. This can be done simply by taking a single 
% time point at the peak of response, averaging over some time window or 
% fitting a hemodynamic response function and computing the magnitude of 
% that. For this tutorial, we will assume that you have a single scalar 
% response for each voxel for each trial already computed. 

% In the forward encoding, one solves for the weight matrix typically by
% doing a least-squares regression. In inverse encoding, one uses this 
% estimated weight matrix to invert the equation and solve for the channel 
% responses.

%% Make simulated data
% First let's simulate some data. Later you will see this done on real data,
% but it's *always* a good idea to run your analyses on simulations where
% you know the ground truth to test how your model behaves. So, let's start
% with some basic assumptions about how the model works. The model assumes 
% that each voxel (or EEG electrode or whatever measurement you are making)
% is a linear sum of individual channels that are tuned for some stimulus 
% feature. We will do this here with orientation as the feature, but this 
% can and has been done with other features (direction of motion, space or 
% really anything in which you can define a tuning function that is 
% reasonable given what you know about the neural representation).

% Start, by making a matrix called neuralResponse in which you have 90 
% neurons (one per row of the matrix) and in each column it's ideal 
% response to one of 180 orientations. Let's assume that the neurons 
% are tuned according to a Von Mises function which is often used as 
% a circular equivalent to a Gaussian. Each neuron will be tuned to a
% different orientation. Why 90 neurons? Just so that we can keep the 
% matrix dimensions clear. Half the battle with linear algebra is getting 
% your matrix dimensions right after all.

% some variables

iNeuron = 1;
orientations = 0:179;
% Notice that there is a parameter called k which is the concentration
% parameter. The larger the number, the narrower the function. 
% A value of around 10 should give something reasonable. Also, 
% don't worry about the denominator part with the Bessel function - 
% that's just a way to normalize the function to have area 1 for a 
% probability distribution, we can just ignore that and normalize to a 
% max height of 1 instead
k = 10;
% loop over each neuron tuning function
for orientPreference = 0:2:179  
  % compute the neural response as a Von Mises function
  % Note the 2 here which makes it so that our 0 - 180 orientation
  % space gets mapped to all 360 degrees
  neuralResponse(iNeuron,:) = exp(k*cos(2*pi*(orientations-orientPreference)/180));
  % normalize to a height of 1
  neuralResponse(iNeuron,:) = neuralResponse(iNeuron,:) / max(neuralResponse(iNeuron,:));
  % update counter
  iNeuron = iNeuron + 1;
end
 
% plot the response of neuron 45
figure;
plot(orientations,neuralResponse(45,:));
xlabel('Orientation');
ylabel('Channel response (normalized units to 1)');
title('Neuron 45')

% plot the response of neuron 25
figure;
plot(orientations,neuralResponse(25,:));
xlabel('Orientation');
ylabel('Channel response (normalized units to 1)');
title('Neuron 25')

% Ok. Now we want to simulate v voxels (say 50) response as random 
% combinations of the neural tuning functions. The basic idea is that 
% each voxel contains some random sub-populations of neurons tuned for 
% different orientations and the total response is just a linear 
% combination of these. So, let's make a random matrix that are the 
% weights of each of these neurons onto each voxel. This should be a 
% matrix called neuronToVoxelWeights that is nNeurons x nVoxels in 
% size where each column contains ranodom weights for each voxel.
  
% make a random weighting of neurons on to each voxel
nNeurons = size(neuralResponse,1);
nVoxels = 50;
neuronToVoxelWeights = rand(nNeurons,nVoxels);

% Now, let's simulate an experiment in which we have nStimuli of 
% different orientations. To keep things simple, we will have 
% nStimuli=8 stimuli that are evenly spaced across orientations 
% starting at 0. And we will have nRepeats (say 20) of each stimuli. 
% In a real experiment, we would randomize the order of the stimuli 
% so as to avoid adaptation and other non-stationary effects, but 
% here we can just have them in a sequential order. We can start by 
% making an array stimuli of length nStimuli x nRepeats with the 
% stimulus values.
 
% make stimulus array
nStimuli = 8;
% evenly space stimuli
stimuli = 0:180/(nStimuli):179;
% number of repeats
nRepeats = 20;
stimuli = repmat(stimuli,1,nRepeats);

% A few simple things here - let's round all the stimulus values 
% to the nearest integer degree (just for ease of calculation) 
% and add one (because Matlab indexes starting with one and not zero)
% and make this into a column array

% round and make a column array
stimuli = round(stimuli(:))+1;

% ok, now we can compute the response to each stimulus. So we should
% make a voxelResponse matrix (dimensions nTrials = nStimuli x nRepeats 
% by nVoxels).

% compute the voxelResponse
nTrials = nStimuli * nRepeats;
for iTrial = 1:nTrials
  % get the neural response to this stimulus, by indexing the correct column of the neuralResponse matrix
  thisNeuralResponse = neuralResponse(:,stimuli(iTrial));
  % multiply this by the neuronToVoxelWeights to get the voxel response on this trial. Note that you need
  % to get the matrix dimensions right, so transpose is needed on thisNeuralResponse
  voxelResponse(iTrial,:) = thisNeuralResponse' * neuronToVoxelWeights;
end
 
% plot the voxelResponse for the 7th trial
figure;
plot(voxelResponse(7,:));
xlabel('Voxel (number)');
ylabel('Voxel response (fake measurement units)');
 
% plot another trial voxel response
figure;
plot(voxelResponse(7,:),'b-.');
hold on
plot(voxelResponse(7+nStimuli,:),'r-o');
xlabel('Voxel (number)');
ylabel('Voxel response (fake measurement units)');

% Ok. Let's fix that by adding random gaussian noise to the voxelResponses
% (BTW - this is a a very simple noise model called IID - independent, 
% identically distributed gaussian noise - in reality, noise is going to 
% have some complex characteristics, for example, there might be more 
% correlation between neighboring voxels than voxels spatially distant - 
% or more correlation between voxels that receive similarly tuning - but 
% for this purpose, let's go with this simple nosie model). Just to keep 
% the scale understandable, let's normalize the voxel responses to have a 
% mean of 1 and then add gaussian noise with a fixed noiseStandardDeviation 
% of, say, 0.05.

% add noise to the voxel responses
noiseStandardDeviation = 0.05;
% normalize response 
voxelResponse = voxelResponse / mean(voxelResponse(:));
% add gaussian noise
voxelResponse = voxelResponse + noiseStandardDeviation*randn(size(voxelResponse));
 
% check the voxelResponses
figure;
stim1 = 7;
stim2 = 3;
subplot(1,3,1);
plot(voxelResponse(stim1,:),'b-.');
hold on
plot(voxelResponse(stim1+nStimuli,:),'r-o');
xlabel('Voxel (number)');
ylabel('Voxel response (fake measurement units)');
 
subplot(1,3,2);
plot(voxelResponse(stim1,:),voxelResponse(stim1+nStimuli,:),'k.');
xlabel('Response to first presentation');
ylabel('Response to second presentation');
axis square
 
subplot(1,3,3);
plot(voxelResponse(stim1,:),voxelResponse(stim2,:),'k.');
xlabel(sprintf('Response to stimulus: %i deg',stimuli(stim1)));
ylabel(sprintf('Response to stimulus: %i',stimuli(stim2)));
axis square

% Okey, dokey. We now have a simulated voxelResponse that we can test 
% our model with. Remember, that everything that we did in the simulation 
% is an assumption: Von Mises tuned neurons, random linear weighting, 
% IID noise and SNR (signal-to-noise ratio as in the magnitude of the 
% orientation tuned response compared to the noise). Each of these may 
% or may not be valid for real data and should ideally be tested (or at 
% least thought about deeply!). The beauty of a simulation is that we 
% can change these assumptions and see how they affect the analysis -
% something that is really, really important to do as you learn about 
% new analysis techniques. If the assumptions are incorrect, the analysis 
% can fall apart in ways that are often unexpected and you may infer 
% incorrect conclusions, so play around with these simulations first to 
% understand what is going on!
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make encoding model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ok, let's build the encoding model. In this case we are going to assume 
% that there are nChannels = 8, that are tuned as exponentiated and 
% rectified sinusoids to different orientations. Note that this is 
% different from the assumptions above of Von Mises tuned neurons. 
% Will get back to that later! Let's build a matrix called channelBasis 
% which is 180 x nChannels that contains the ideal channel responses 
% (also called channel basis functions) to each of 180 orientations. 
% We will use gaussian functions raised to the exponent 7.

% make channel basis functions
nChannels = 8;
exponent = 7;
orientations = 0:179;
prefOrientation = 0:180/nChannels:179;

% loop over each channel
for iChannel = 1:nChannels
  % get sinusoid. Note the 2 here which makes it so that our 0 - 180 orientation
  % space gets mapped to all 360 degrees
  thisChannelBasis =  cos(2*pi*(orientations-prefOrientation(iChannel))/180);
  % rectify
  thisChannelBasis(thisChannelBasis<0) = 0;
  % raise to exponent
  thisChannelBasis = thisChannelBasis.^exponent;
  % keep in matrix
  channelBasis(:,iChannel) = thisChannelBasis;
end
 
% plot channel basis functions
figure;
plot(orientations,channelBasis);
xlabel('Preferred orientation (deg)');
ylabel('Ideal channel response (normalized to 1)');
 
% Great. Now, let's compute responses of these idealized channel 
% basis functions to each one of th nTrials in the array stimuli. 
% We will compute a nTrial x nChannel matrix called channelResponse

% compute the channelResponse for each trial
for iTrial = 1:nTrials
  channelResponse(iTrial,:) = channelBasis(stimuli(iTrial),:);
end
 
% Easy, right? Now let's fit this model to the simulated data. 
% Remember that the model that we have is: channelResponses 
% (nTrials x nChannels) x estimatedWeights (nChannels x nVoxels) = voxelResponses (nTrials x nVoxels) 
% You can solve this by doing your favorite least-squares estimation 
% procedure (or if you want to be fancy, you could do some robust 
% regression technique). We'll just do the basic here.

% compute estimated weights
estimatedWeights =  pinv(channelResponse) * voxelResponse;

% Whenever you fit a model, it's always important to compute a measure
% of model fit - how well does the model actually fit the data. r2, 
% amount of variance accounted for, is a good measure for this sort of 
% data. So, compute that by seeing what the model predicts for the data 
% (that's easy, that's just channelResponse x estimatedWeights and remove
% that from the data. Then compute 1 minus the residual variance / variance.

% compute model prediction
modelPrediction = channelResponse * estimatedWeights;
% compute residual
residualResponse = voxelResponse-modelPrediction;
% compute r2
r2 = 1-var(residualResponse(:))/var(voxelResponse(:))
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inverted encoding model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ok. Now, let's invert the model to see what it predicts for channel 
% responses. But, first we gotta talk about cross-validation. Always, 
% always cross-validate. There, we are done talking about it. Here, let's 
% do a simple version where we split the data into two (split-half 
% cross-validation). Make two matrices from your voxelData where 
% trainVoxelResponse is the first half trials and testVoxelResponse is 
% the second half. Then compute estimatedWeights on the trainVoxelResponse.

% split half into train and test
firstHalf = 1:round(nTrials/2);
secondHalf = round(nTrials/2)+1:nTrials;
trainVoxelResponse = voxelResponse(firstHalf,:);
testVoxelResponse = voxelResponse(secondHalf:end,:);
% compute weights on train data
estimatedWeights = pinv(channelResponse(firstHalf,:))*trainVoxelResponse;

% Now on the second half of the data, compute the estimatedChannelResponse
% (look at above equations for channelResponses and use least-squares 
% estimation

% compute channel response from textVoxelResponses
estimatedChannelResponse = testVoxelResponse * pinv(estimatedWeights);

% Plot the mean estimatedChannelResponse for each stimulus type:
figure;colors = hsv(nStimuli);
for iStimuli = 1:nStimuli
  plot(prefOrientation,mean(estimatedChannelResponse(iStimuli:nStimuli:end,:),1),'-','Color',colors(iStimuli,:));
  hold on
end
xlabel('Channel orientation preference (deg)');
ylabel('Estimated channel response (percentile of max)');
title(sprintf('r2 = %0.4f',r2));
 
% Maybe, a little too pretty. Try the whole simulation again, but this 
% time add more noise (try, say an order of magnitude larger noise standard 
% deviation - 0.5) and see what happens to the estimated channel response 
% profiles. Make sure to keep the original low noise voxelResponses by 
% naming the new voxelResponses something different like voxelResponseNoisy

% Compute voxel response without noise
nTrials = nStimuli * nRepeats;
for iTrial = 1:nTrials
  % get the neural response to this stimulus, by indexing the correct column of the neuralResponse matrix
  thisNeuralResponse = neuralResponse(:,stimuli(iTrial));
  % multiply this by the neuronToVoxelWeights to get the voxel response on this trial. Note that you need
  % to get the matrix dimensions right, so transpose is needed on thisNeuralResponse
  voxelResponseNoisy(iTrial,:) = thisNeuralResponse' * neuronToVoxelWeights;
end
 
% add noise
noiseStandardDeviation = 0.5;
% normalize response 
voxelResponseNoisy = voxelResponseNoisy / mean(voxelResponseNoisy(:));
% add gaussian noise
voxelResponseNoisy = voxelResponseNoisy + noiseStandardDeviation*randn(size(voxelResponseNoisy));
 
% split into train and tes
trainVoxelResponseNoisy = voxelResponseNoisy(firstHalf,:);
testVoxelResponseNoisy = voxelResponseNoisy(secondHalf:end,:);
 
% compute weights on train data
estimatedWeights = pinv(channelResponse(firstHalf,:))*trainVoxelResponseNoisy;
 
% compute model prediction on test data
modelPrediction = channelResponse(secondHalf,:) * estimatedWeights;

% compute residual
residualResponse = testVoxelResponseNoisy-modelPrediction;
% compute r2
r2 = 1-var(residualResponse(:))/var(testVoxelResponseNoisy(:))
 
% invert model and compute channel response
estimatedChannelResponse = testVoxelResponseNoisy * pinv(estimatedWeights);
 
% plot estimated channel profiles
figure;colors = hsv(nStimuli);
for iStimuli = 1:nStimuli
  plot(prefOrientation,mean(estimatedChannelResponse(iStimuli:nStimuli:end,:),1),'-','Color',colors(iStimuli,:));
  hold on
end
xlabel('Channel orientation preference (deg)');
ylabel('Estimated channel response (percentile of max)');
title(sprintf('r2 = %0.4f',r2));
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stimulus likelihood function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% split half into train and test
firstHalf = 1:round(nTrials/2);
secondHalf = round(nTrials/2)+1:nTrials;
trainVoxelResponse = voxelResponse(firstHalf,:);
testVoxelResponse = voxelResponse(secondHalf:end,:);
 
% compute weights on train data
estimatedWeights = pinv(channelResponse(firstHalf,:))*trainVoxelResponse;
 
% compute model prediction on test data
modelPrediction = channelResponse(secondHalf,:) * estimatedWeights;
% compute residual
residualResponse = testVoxelResponseNoisy-modelPrediction;
% compute residual variance, note that this is a scalar
residualVariance = var(residualResponse(:));
 
% make this into a covariance matrix in which the diagonal contains the variance for each voxel
% and off diagonals (in this case all 0) contain covariance between voxels
modelCovar = eye(nVoxels)*residualVariance;
 
% cycle over each trial
nTestTrials = size(testVoxelResponse,1);
for iTrial = 1:nTestTrials
  % now cycle over all possible orientation
  for iOrientation = 1:179
    % compute the mean voxel response predicted by the channel encoding model
    predictedResponse = channelBasis(iOrientation,:)*estimatedWeights;
    % now use that mean response and the model covariance to estimate the probability
    % of seeing this orientation given the response on this trial
    likelihood(iTrial,iOrientation) = mvnpdf(testVoxelResponse(iTrial,:),predictedResponse,modelCovar);
  end
end
 
figure
for iStimuli = 1:nStimuli
  plot(1:179,mean(likelihood(iStimuli:nStimuli:end,:),1),'-','Color',colors(iStimuli,:));
  hold on
end
xlabel('stimulus orientation (deg)');
ylabel('probability given trial response');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inverted Encoding model with different channel basis functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reweight the channels
channelReweighting = [0 0.8 0.4 0 0 0 0.4 0.8]';
% make into a full matrix xform to transform the original channels
for iChannel = 1:nChannels
  xform(iChannel,:) = circshift(channelReweighting,iChannel-1);
end
% and get new bimodal channels
bimodalChannelBasis = channelBasis * xform;
 
% display a figure with one of the channels
figure
plot(orientations,bimodalChannelBasis(:,5));
xlabel('orientation (deg)');
ylabel('Channel response (normalized to 1)');
 
% compute the channelResponse for each trial
for iTrial = 1:nTrials
  channelResponse(iTrial,:) = bimodalChannelBasis(stimuli(iTrial),:);
end
 
% compute estimated weights
estimatedWeights =  pinv(channelResponse) * voxelResponse;
 
% compute model prediction
modelPrediction = channelResponse * estimatedWeights;
% compute residual
residualResponse = voxelResponse-modelPrediction;
% compute r2
r2 = 1-var(residualResponse(:))/var(voxelResponse(:))
 
% compute estimated channel response profiles
estimatedChannelResponse = testVoxelResponse * pinv(estimatedWeights);
 
% and plot one of the channels averaged across all trials
figure;
plot(prefOrientation,mean(estimatedChannelResponse(5:nStimuli:end,:),1));
xlabel('Channel preferred orientation (deg)');
ylabel('Estimated channel response (percentile of full)');
title(sprintf('r2 = %0.4f',r2));
 
% compute weights on train data
estimatedWeights = pinv(channelResponse(firstHalf,:))*trainVoxelResponse;
 
% compute model prediction on test data
modelPrediction = channelResponse(secondHalf,:) * estimatedWeights;
% compute residual
residualResponse = testVoxelResponseNoisy-modelPrediction;
% compute residual variance, note that this is a scalar
residualVariance = var(residualResponse(:));
 
% make this into a covariance matrix in which the diagonal contains the variance for each voxel
% and off diagonals (in this case all 0) contain covariance between voxels
modelCovar = eye(nVoxels)*residualVariance;
 
% cycle over each trial
nTestTrials = size(testVoxelResponse,1);
for iTrial = 1:nTestTrials
  % now cycle over all possible orientation
  for iOrientation = 1:179
    % compute the mean voxel response predicted by the channel encoding model
    predictedResponse = bimodalChannelBasis(iOrientation,:)*estimatedWeights;
    % now use that mean response and the model covariance to estimate the probability
    % of seeing this orientation given the response on this trial
    likelihood(iTrial,iOrientation) = mvnpdf(testVoxelResponse(iTrial,:),predictedResponse,modelCovar);
  end
end
 
% now plot the likelihood function averaged over repeats
figure
for iStimuli = 1:nStimuli
  plot(1:179,mean(likelihood(iStimuli:nStimuli:end,:),1));
  hold on
end
xlabel('stimulus orientation (deg)');
ylabel('probability given trial response');


