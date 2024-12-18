% Load the data
data = load('../../data/ilen_stor_17nov.mat');
data = data.downsampledFrames;
%data = data(:, :, 138:end); %this is for the 270 DS variant

% Get dimensions
nx = size(data, 1);
ny = size(data, 2);
n = nx * ny;
m = size(data, 3);

% Reshape and convert to double
data_reshaped = reshape(data, [nx * ny, m]);
%data_reshaped = data_reshaped - mean(data_reshaped, 2);
data_reshaped = double(data_reshaped);

%% Normalization

% Min-Max Normalization per column
min_vals = min(data_reshaped, [], 1);
max_vals = max(data_reshaped, [], 1);
epsilon = 1e-8; % To avoid division by zero
data_normalized = (data_reshaped - min_vals) ./ (max_vals - min_vals + epsilon);

%% Partition Data

addpath ../../utils
% Partition normalized data into training and test sets
mTrain = 0.9 * m;
[Train, Test, flow.mean_flow, ~] = partition(data_normalized, mTrain);
disp(size(Train))

% RMS vorticity in the training set - used for rescaling
flow.avg_energy = mean( sqrt( mean(Train.^2, 1) ) );

