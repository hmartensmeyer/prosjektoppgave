%% Load cylinder data files and useful parameters
% Jared Callaham 2018

data = load('../../data/vortexShedding.mat');

%% Calculate velocity fields

% Step 1: Initialize matrices to hold the velocity components and magnitude for all timesteps
num_timesteps = 600;
u_field = zeros(223, 196, num_timesteps);
v_field = zeros(223, 196, num_timesteps);
velocity_magnitude_field = zeros(223, 196, num_timesteps);

% Step 2: Loop through the timesteps, extract u and v, calculate the magnitude, and store them
for t = 1:num_timesteps
    u = data.velocityField(t).u;  % Extract u for timestep t
    v = data.velocityField(t).v;  % Extract v for timestep t
    
    % Store u and v in their respective fields
    u_field(:,:,t) = u;
    v_field(:,:,t) = v;
    
    % Calculate the velocity magnitude at each point
    velocity_magnitude_field(:,:,t) = sqrt(u.^2 + v.^2);
end

% Step 3: Reshape the fields to 2D matrices for further processing or visualization
u_field_reshaped = reshape(u_field, [size(u_field, 1) * size(u_field, 2), size(u_field, 3)]);
v_field_reshaped = reshape(v_field, [size(v_field, 1) * size(v_field, 2), size(v_field, 3)]);
velocity_magnitude_field_reshaped = reshape(velocity_magnitude_field, [size(velocity_magnitude_field, 1) * size(velocity_magnitude_field, 2), size(velocity_magnitude_field, 3)]);

% Display sizes of the reshaped fields
disp(size(u_field_reshaped));
disp(size(v_field_reshaped));
disp(size(velocity_magnitude_field_reshaped));

%%

nx = 223; ny = 196;
n = nx*ny;  % y is streamwise direction, x is cross-stream
m = 600;  % Total number of available snapshots

mTrain = 600;  % Number of snapshots to use for training (one full period of vortex shedding)

addpath ../utils
% Partition data into training and test set. Also returns the mean flow and any masked values
%[Train, Test, flow.mean_flow, ~] = partition(u_field_reshaped, mTrain);
%disp(size(Train))

flow.mean_flow = mean(flow.mean_flow);

% RMS vorticity in the training set - used for rescaling
%flow.avg_energy = mean( sqrt( mean(Train.^2, 1) ) );

% % Helpful for plotting
% load('../utils/CCcool.mat')
% flow.cmap = CC;
% flow.clim = [-5, 5];
