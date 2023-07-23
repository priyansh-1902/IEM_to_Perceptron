# Tracking Covert Spatial Attention using Linear and Non Linear Perceptrons

## Basic Overview

Covert Visual attention can be tracked by alpha-band activity. [[1]](#1) Inverted Encoding Models have been used before to track spatial attention. <br>
We use Non Linear Perceptrons to do the same.

### Channel Tuning Functions

To each spatial location we associate a tuning function. For the experimental data presented in [[1]](#1), we have 8 locations and therefore 8 corresponding channels (Fig 1a.). <br> 
The channel tuning function for each of these channels is 

<p align="center">
  <img src="./Figures/Rolled Channel Tuning Functions.jpeg" alt="Rolled Channel Tuning Functions" width="325">
  <img src="./Figures/Channel Tuning Function.jpeg" alt="Channel Tuning Function" width="325">
</p>

### Inverted Encoding Models (Forward Computation + Inversion of Weight Matrix)
$B_1 \rightarrow$ EEG Matrix ($n$ electrodes $\times$ $m$ trials) <br>
$C_1 \rightarrow$ Channel Tuning Functions ($k$ channels $\times$ $m$ trials)
### Non Linear Perceptrons (Forward Computation + Back Propogation to compute weights)



## Dependencies
```
python>=3.7
PyTorch
numpy
seaborn, matplotlib
Matlab R2023b + Signal Processing Toolbox
```

## Installation



## Downloading EEG and Behaviour Data

## How to use

To visualize spatial attention as heat maps
```
python run.py --model LinearPerceptron --numIterations 10 --startTime 0 --endTime 600 --verbose False --saveHeatMap "./trial.jpeg"
```

## References
<a id="1">[1]</a> 
Foster et al. (2017). 
Alpha-Band Oscillations Enable Spatially and Temporally Resolved Tracking of Covert Spatial Attention
