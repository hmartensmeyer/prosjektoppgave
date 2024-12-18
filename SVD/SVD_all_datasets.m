% Improved MATLAB code: Processing and SVD for multiple datasets

% Initialize structures to store the results
singular_values_all = struct();
normalized_singular_values_all = struct();
cumulative_energy_all = struct();
U_all = struct();
S_all = struct();
V_all = struct();
r_90_all = struct();

%% Vortex shedding data (Re = 100)
% Load data
load('../../data/CYLINDER_ALL.mat');
data_U = UALL;
data_V = VALL;
data_VS = sqrt(data_U.^2 + data_V.^2);

% Process data
[singular_values_VS, normalized_singular_values_VS, cumulative_energy_VS, U_VS, S_VS, V_VS] = process_data(data_VS, 2);

% Save results
singular_values_all.VS = singular_values_VS;
normalized_singular_values_all.VS = normalized_singular_values_VS;
cumulative_energy_all.VS = cumulative_energy_VS;
r_90_all.VS = find(cumulative_energy_VS >= 0.9, 1);
U_all.VS = U_VS;
S_all.VS = S_VS;
V_all.VS = V_VS;
disp('Here')

% %% Sea Surface Temperature (SST)
% % Load data
% file_SST = load('../../data/sst_weekly.mat');
% data_SST = file_SST.sst;
% 
% % Check for NaNs and Infinities
% data_SST = fillmissing(data_SST, 'constant', 0);
% data_SST(isinf(data_SST)) = 0;
% 
% % Process data
% [singular_values_SST, normalized_singular_values_SST, cumulative_energy_SST, U_SST, S_SST, V_SST] = process_data(data_SST, 2);
% 
% % Save results
% singular_values_all.SST = singular_values_SST;
% normalized_singular_values_all.SST = normalized_singular_values_SST;
% cumulative_energy_all.SST = cumulative_energy_SST;
% r_90_all.SST = find(cumulative_energy_SST >= 0.9, 1);
% U_all.SST = U_SST;
% S_all.SST = S_SST;
% V_all.SST = V_SST;
% disp('Here')
%% DNS test data, case1, surface elevation
% Load data
data_DNS = load('../data/eta.mat');
data_DNS = data_DNS.eta;

% Reshape the data
data_DNS = reshape(data_DNS, 5000, 256*256);

% Normalization

% Min-Max Normalization per column
min_vals = min(data_DNS, [], 1);
max_vals = max(data_DNS, [], 1);
epsilon = 1e-8; % To avoid division by zero
data_DNS = (data_DNS - min_vals) ./ (max_vals - min_vals + epsilon);

% Process data
[singular_values_DNS, normalized_singular_values_DNS, cumulative_energy_DNS, U_DNS, S_DNS, V_DNS] = process_data(data_DNS, 2);

% Save results
singular_values_all.DNS = singular_values_DNS;
normalized_singular_values_all.DNS = normalized_singular_values_DNS;
cumulative_energy_all.DNS = cumulative_energy_DNS;
r_90_all.DNS = find(cumulative_energy_DNS >= 0.9, 1);
U_all.DNS = U_DNS;
S_all.DNS = S_DNS;
V_all.DNS = V_DNS;
disp('Here')
% %% DNS test data, case1, velocity
% % Load data
% data_DNS_U = h5read('../../data/u.mat', '/uSurface');
% 
% % Reshape the data
% data_DNS_U = reshape(data_DNS_U, 5000, 256*256);
% 
% % Process data
% [singular_values_DNS_U, normalized_singular_values_DNS_U, cumulative_energy_DNS_U, U_DNS_U, S_DNS_U, V_DNS_U] = process_data(data_DNS_U, 2);
% 
% % Save results
% singular_values_all.DNS_U = singular_values_DNS_U;
% normalized_singular_values_all.DNS_U = normalized_singular_values_DNS_U;
% cumulative_energy_all.DNS_U = cumulative_energy_DNS_U;
% r_90_all.DNS_U = find(cumulative_energy_DNS_U >= 0.9, 1);
% U_all.DNS_U = U_DNS_U;
% S_all.DNS_U = S_DNS_U;
% V_all.DNS_U = V_DNS_U;
% disp('Here')
%% Vortex shedding data (Re = 7000)
% Load data
data = load('../../data/vortexShedding.mat');

% Calculate velocity magnitude field
velocity_magnitude_field = zeros(223, 196, 600);

for t = 1:600
    u = data.velocityField(t).u;
    v = data.velocityField(t).v;
    velocity_magnitude_field(:,:,t) = sqrt(u.^2 + v.^2);
end

% Reshape and transpose the data
data_VS_higher_Re = reshape(velocity_magnitude_field, [223*196, 600])';
%%
% Process data
[singular_values_VS_higher_Re, normalized_singular_values_VS_higher_Re, cumulative_energy_VS_higher_Re, U_VS_higher_Re, S_VS_higher_Re, V_VS_higher_Re] = process_data(data_VS_higher_Re, 2);

% Save results
singular_values_all.VS_higher_Re = singular_values_VS_higher_Re;
normalized_singular_values_all.VS_higher_Re = normalized_singular_values_VS_higher_Re;
cumulative_energy_all.VS_higher_Re = cumulative_energy_VS_higher_Re;
r_90_all.VS_higher_Re = find(cumulative_energy_VS_higher_Re >= 0.9, 1);
U_all.VS_higher_Re = U_VS_higher_Re;
S_all.VS_higher_Re = S_VS_higher_Re;
V_all.VS_higher_Re = V_VS_higher_Re;
disp('Here')
% %% Vortex shedding data (Re = 7000), with noise
% % Add salt & pepper noise
% X = data_VS_higher_Re;
% eta = 0.1; % Fraction of occluded points
% 
% if eta ~= 0
%     rep = std(X(:)) * 10;
%     x = rand(size(X));
%     b = sort(x(:));
%     thresh = b(floor(.5 * eta * numel(b)));
%     X(x < thresh) = rep;
%     x = rand(size(X));
%     b = sort(x(:));
%     thresh = b(floor(.5 * eta * numel(b)));
%     X(x < thresh) = -rep;
% end
% 
% data_VS_higher_Re_noise = X;
% 
% % Process data
% [singular_values_VS_higher_Re_noise, normalized_singular_values_VS_higher_Re_noise, cumulative_energy_VS_higher_Re_noise, U_VS_higher_Re_noise, S_VS_higher_Re_noise, V_VS_higher_Re_noise] = process_data(data_VS_higher_Re_noise, 1);
% 
% % Save results
% singular_values_all.VS_higher_Re_noise = singular_values_VS_higher_Re_noise;
% normalized_singular_values_all.nidelva = normalized_singular_values_VS_higher_Re_noise;
% cumulative_energy_all.VS_higher_Re_noise = cumulative_energy_VS_higher_Re_noise;
% r_90_all.VS_higher_Re_noise = find(cumulative_energy_VS_higher_Re_noise >= 0.9, 1);
% U_all.VS_higher_Re_noise = U_VS_higher_Re_noise;
% S_all.VS_higher_Re_noise = S_VS_higher_Re_noise;
% V_all.VS_higher_Re_noise = V_VS_higher_Re_noise;
% disp('Here')

%%

data = load("../data/ilen_stor_17nov.mat");

raw_data = data.downsampledFrames;
reshaped_data = reshape(raw_data, [540*540, 1058]); %shouldn't hard code this man
%reshaped_data = reshaped_data(:, 1:100);
reshaped_data = double(reshaped_data);

% Normalization of SVD-recon

% Min-Max Normalization per column
min_vals = min(reshaped_data, [], 1);
max_vals = max(reshaped_data, [], 1);
epsilon = 1e-8; % To avoid division by zero
reshaped_data = (reshaped_data - min_vals) ./ (max_vals - min_vals + epsilon);
%%
[singular_values_nidelva, normalized_singular_values_nidelva, cumulative_energy_nidelva, U, S, V] = process_data(reshaped_data, 2);

% Save results
singular_values_all.nidelva = singular_values_nidelva;
normalized_singular_values_all.nidelva = normalized_singular_values_nidelva;
cumulative_energy_all.nidelva = cumulative_energy_nidelva;
r_90_all.nidelva = find(cumulative_energy_nidelva >= 0.9, 1);
U_all.nidelva = U;
S_all.nidelva = S;
V_all.nidelva = V;
disp('Nidelva done')

%% Save SVD results to a MAT-file
save('output/svd_results.mat', 'singular_values_all', 'normalized_singular_values_all', 'cumulative_energy_all', 'r_90_all'); %, 'U_all', "S_all", "V_all", '-v7.3');

%%

% Function to process data and compute SVD and energy
function [singular_values, normalized_singular_values, cumulative_energy, U, S, V] = process_data(data, mean_dim)
    % Compute mean along specified dimension
    mean_data = mean(data, mean_dim);

    % Subtract mean from data
    data_centered = bsxfun(@minus, data, mean_data);

    % Perform SVD
    [U, S, V] = svd(data_centered, 'econ');

    % Calculate singular values, normalized singular values, cumulative energy
    singular_values = diag(S);
    normalized_singular_values = singular_values / sum(singular_values);
    cumulative_energy = cumsum(singular_values) / sum(singular_values);
end