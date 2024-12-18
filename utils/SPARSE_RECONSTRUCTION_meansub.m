function [Reconstructed_Flows, residuals, sims, sensor_pos] = SPARSE_RECONSTRUCTION_meansub(data, flow, num_timesteps, num_repeats, ns, sensor_type, SF, rescale, nx, ny)
%RECONSTRUCT_FLOWS Reconstructs flow fields from measurements using QR sensors.
%   This function reconstructs flow fields from sensor measurements using a
%   dictionary built from training data and QR decomposition to select sensor
%   locations.
%
%   Inputs:
%       data_normalized - Normalized data matrix of flow fields (n x m)
%       flow            - Additional flow information required by sp_approx and reconstruct functions
%       num_timesteps   - Number of timesteps to process (default: 50)
%       num_repeats     - Number of repeats per timestep for averaging (default: 1)
%       ns              - Number of sensors to use (default: 30)
%       QR              - Choose type of sensors
%       SF              - sampling frequency
%
%   Outputs:
%       Reconstructed_Flows - Reconstructed flow fields (n x num_timesteps)
%       residuals           - Residuals for each timestep (num_timesteps x 1)
%       sims                - Similarity measures for each timestep (num_timesteps x 1)

    % Check and set default values
    if nargin < 3 || isempty(num_timesteps)
        num_timesteps = 50;
    end

    if nargin < 4 || isempty(num_repeats)
        num_repeats = 1;
    end

    if nargin < 5 || isempty(ns)
        ns = 30;
    end

    % Get the number of spatial points (n) and total timesteps (m)
    [n, m] = size(data);

    % get mean
    mean_flow = (mean(data, 2));

    % Ensure num_timesteps does not exceed available data
    num_timesteps = min(num_timesteps, m);

    % Define the training data by selecting every 10th column
    Train = data(:, 1:SF:end) - mean_flow;

    switch sensor_type
        case 'qr'
            % Perform QR decomposition with column pivoting on the transpose of Train
            [~, ~, E] = qr(Train', 'vector');
            E = E';  % Transpose to make it a column vector
            % Sensor indices selected via QR-factorization
            linear_idx_qr = E(1:ns);
            % Construct the measurement matrix C
            C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
            sensor_pos = linear_idx_qr;
        case 'random'
            % % Randomly select sensor indices from the flattened grid
            % sensor_linear_idx = randsample(n, ns);
            % % Convert linear indices to (row, column) indices
            % [sensor_y, sensor_x] = ind2sub([ny, nx], sensor_linear_idx);

            % Restrict to cylinder wake: final 80% of width, middle 50% of height
            sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
            sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake

            % Combine into a single array for easy access to sensor locations
            %sensor_idx = [sensor_y, sensor_x];
    
            % Convert to measurement matrix
            C = spdiags(ones(n, 1), 0, n, n);
            C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
            sensor_pos = sensor_idx;
    end

    % Initialize matrices to store reconstructed flow fields, residuals, and similarities
    Reconstructed_Flows = zeros(n, num_timesteps);
    residuals = zeros(num_timesteps, 1);
    sims = zeros(num_timesteps, 1);

    for t = 1:num_timesteps
        % True flow field at timestep t
        x = data(:, t) - mean_flow;

        % Initialize variables to accumulate results over repeats
        x_hat_sum = zeros(n, 1);  % Sum of reconstructed flow fields
        res_sum = 0;              % Sum of residuals
        sim_sum = 0;

        for repeat = 1:num_repeats
            % Sensor measurements
            y = C * x;

            % Measured dictionary
            D = C * Train;

            % Sparse approximation
            eta = 0;
            s = sp_approx(y, D, eta, flow);

            % Reconstruct the flow field
            [x_hat, res, sim] = reconstruct(x, Train, s, mean_flow, false, true);

            % Accumulate reconstructed flow fields and residuals
            x_hat_sum = x_hat_sum + x_hat;
            res_sum = res_sum + res;
            sim_sum = sim_sum + sim;
        end

        % Compute average reconstructed flow field and residual for timestep t
        x_hat_avg = x_hat_sum / num_repeats;
        res_avg = res_sum / num_repeats;
        sim_avg = sim_sum / num_repeats;

        % Add mean flow back to the reconstructed flow field
        x_hat_avg_full = x_hat_avg + mean_flow;

        % Store the averaged reconstructed flow field and residual
        Reconstructed_Flows(:, t) = x_hat_avg_full;
        residuals(t) = res_avg;
        sims(t) = sim_avg;

        % Display progress
        disp(['_____________________________________________________________________________________Timestep ', num2str(t), ': Average residual over ', num2str(num_repeats), ' repeats = ', num2str(res_avg), '   SSIM = ', num2str(sim_avg)]);
    end
end
