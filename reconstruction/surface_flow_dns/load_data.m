%% Load cylinder data files and useful parameters
% Jared Callaham 2018

data = load('../../data/eta.mat');

%%
nx = 256; ny = 256;
n = nx*ny;  % y is streamwise direction, x is cross-stream
m = 5000;  % Total number of available snapshots

mTrain = 5000;  % Number of snapshots to use for training (one full period of vortex shedding)

data = data.eta;
data_reshaped = reshape(data, [nx*ny, m]);

%% Normalization

% Min-Max Normalization per column
min_vals = min(data_reshaped, [], 1);
max_vals = max(data_reshaped, [], 1);
epsilon = 0; % To avoid division by zero
data_normalized = (data_reshaped - min_vals) ./ (max_vals - min_vals + epsilon);

% %%
% 
% addpath ../../utils
% % Partition data into training and test set. Also returns the mean flow and any masked values
% [Train, Test, flow.mean_flow, ~] = partition(data_reshaped, mTrain);
% disp(size(Train))
% 
% flow.mean_flow = mean(flow.mean_flow);
% 
% % RMS vorticity in the training set - used for rescaling
% flow.avg_energy = mean( sqrt( mean(Train.^2, 1) ) );
