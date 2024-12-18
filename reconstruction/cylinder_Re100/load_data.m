
load('../../data/CYLINDER_ALL.mat')

% flow.nx = nx; flow.ny = ny;
% flow.n = nx*ny;  % y is streamwise direction, x is cross-stream
% flow.m = size(UALL, 2);  % Total number of available snapshots
ny = 449;
nx = 199;
n = nx*ny;
m = 151;

flow.mTrain = 32;  % Number of snapshots to use for training (one full period of vortex shedding)

addpath ../../utils
% Partition data into training and test set. Also returns the mean flow and any masked values
[Train, Test, flow.mean_flow, ~] = partition(sqrt(UALL.^2 + VALL.^2), flow.mTrain);

% RMS vorticity in the training set - used for rescaling
flow.avg_energy = mean( sqrt( mean(Train.^2, 1) ) );