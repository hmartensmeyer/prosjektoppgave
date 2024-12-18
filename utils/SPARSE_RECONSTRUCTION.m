function [Reconstructed_Flows, residuals, sims, C] = SPARSE_RECONSTRUCTION(data, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny)
%RECONSTRUCT_FLOWS Reconstructs flow fields from measurements using sprase sensors.
%   This function reconstructs flow fields from sensor measurements using a
%   dictionary built from training data and QR decomposition to select sensor
%   locations or random sensors.
%
%   Inputs:
%       data_normalized - Normalized data matrix of flow fields (n x m)
%       flow            - Additional flow information required by sp_approx and reconstruct functions
%       num_timesteps   - Number of timesteps to process (default: 50)
%       num_repeats     - Number of repeats per timestep for averaging (default: 1)
%       ns              - Number of sensors to use (default: 30)
%       sensor_type     - Choose type of sensors
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

    % Ensure num_timesteps does not exceed available data
    num_timesteps = min(num_timesteps, m);

    % Define the training data by selecting every 10th column
    Train = data(:, 1:SF:end);

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
            % Randomly select sensor indices from the flattened grid
            sensor_linear_idx = randsample(n, ns);
            % Convert linear indices to (row, column) indices
            [sensor_y, sensor_x] = ind2sub([ny, nx], sensor_linear_idx);
            % Combine into a single array for easy access to sensor locations
            sensor_idx = [sensor_y, sensor_x];
    
            % Convert to measurement matrix
            C = spdiags(ones(n, 1), 0, n, n);
            C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
            sensor_pos = sensor_idx;
        case 'QR_cluster'
            % Define block size
            block_size = 20;
        
            % Ensure 'ns' (number of sensors/blocks) is defined appropriately
            % ns = 50;  % Uncomment and set as needed
        
            % Perform QR factorization with column pivoting on the transpose of the training data
            [Q, R, p] = qr(Train', 'vector');
        
            % Select top 'ns' sensor indices based on pivoting
            selected_indices = p(1:ns);
        
            % Convert linear indices to (y, x) positions
            [selected_y, selected_x] = ind2sub([ny, nx], selected_indices);
        
            % Initialize measurement matrix 'C' and sensor positions
            C = sparse(ns, n);
            sensor_pos = zeros(ns, 2);
            half_block = floor(block_size / 2);
        
            for i = 1:ns
                center_y = selected_y(i);
                center_x = selected_x(i);
        
                % Define block boundaries around the center
                start_y = max(center_y - half_block + 1, 1);
                start_x = max(center_x - half_block + 1, 1);
                end_y = min(start_y + block_size - 1, ny);
                end_x = min(start_x + block_size - 1, nx);
        
                % Adjust start positions if at the edge
                if end_y - start_y + 1 < block_size
                    start_y = end_y - block_size + 1;
                end
                if end_x - start_x + 1 < block_size
                    start_x = end_x - block_size + 1;
                end
        
                % Get indices of all pixels within the block
                [block_y, block_x] = meshgrid(start_y:end_y, start_x:end_x);
                block_y = block_y(:);
                block_x = block_x(:);
                linear_indices = sub2ind([ny, nx], block_y, block_x);
        
                % Update the measurement matrix 'C' and record sensor positions
                C(i, linear_indices) = 1;
                sensor_pos(i, :) = [start_y, start_x];
            end

            
            % Optional normalization
            % C = C / (block_size^2);

        case 'random_cluster'
            % Define block size and overlap
            block_size = 20;
            overlap = 5;  % Adjust as needed
            
            % Identify ROIs (using edge detection as an example)
            frame = reshape(Train(:,1), ny, nx);  % Example: first frame
            edges = edge(frame, 'Canny');
            heatmap = imgaussfilt(double(edges), 5);
            heatmap = heatmap / max(heatmap(:));
            
            % Number of sensor blocks
            % ns = 50;  % Ensure 'ns' is defined appropriately
            
            % Select top ns block centers based on heatmap
            heatmap_flat = heatmap(:);
            [~, sorted_indices] = sort(heatmap_flat, 'descend');
            [sorted_y, sorted_x] = ind2sub([ny, nx], sorted_indices);
            selected_centers = [];
            min_distance = block_size;
            
            for i = 1:length(sorted_y)
                center = [sorted_y(i), sorted_x(i)];
                if isempty(selected_centers) || all(vecnorm(selected_centers - center, 2, 2) > min_distance)
                    selected_centers = [selected_centers; center];
                    if size(selected_centers, 1) == ns
                        break;
                    end
                end
            end
            
            % Fill remaining blocks randomly if needed
            if size(selected_centers, 1) < ns
                remaining = ns - size(selected_centers, 1);
                excluded_indices = sub2ind([ny, nx], selected_centers(:,1), selected_centers(:,2));
                available_indices = setdiff(1:n, excluded_indices);
                random_selected = randsample(available_indices, remaining);
                [rand_y, rand_x] = ind2sub([ny, nx], random_selected);
                selected_centers = [selected_centers; [rand_y, rand_x]];
            end
            
            % Construct measurement matrix
            C = sparse(ns, n);
            sensor_pos = zeros(ns, 2);
            half_block = floor(block_size / 2);
            
            for i = 1:ns
                center_y = selected_centers(i, 1);
                center_x = selected_centers(i, 2);
                start_y = max(center_y - half_block + 1, 1);
                start_x = max(center_x - half_block + 1, 1);
                end_y = min(start_y + block_size - 1, ny);
                end_x = min(start_x + block_size - 1, nx);
                if end_y - start_y + 1 < block_size
                    start_y = end_y - block_size + 1;
                end
                if end_x - start_x + 1 < block_size
                    start_x = end_x - block_size + 1;
                end
                [block_y, block_x] = meshgrid(start_y:start_y+block_size-1, start_x:start_x+block_size-1);
                block_y = block_y(:);
                block_x = block_x(:);
                linear_indices = sub2ind([ny, nx], block_y, block_x);
                C(i, linear_indices) = 1;
                sensor_pos(i, :) = [start_y, start_x];
            end
        otherwise
            error('Unknown sensor type specified.');
    end

    % Initialize matrices to store reconstructed flow fields, residuals, and similarities
    Reconstructed_Flows = zeros(n, num_timesteps);
    residuals = zeros(num_timesteps, 1);
    sims = zeros(num_timesteps, 1);

    for t = 1:num_timesteps
        % True flow field at timestep t
        x = data(:, t);

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
            [x_hat, res, sim] = reconstruct(x, Train, s, flow, false, false);

            % Accumulate reconstructed flow fields and residuals
            x_hat_sum = x_hat_sum + x_hat;
            res_sum = res_sum + res;
            sim_sum = sim_sum + sim;
        end

        % Compute average reconstructed flow field and residual for timestep t
        x_hat_avg = x_hat_sum / num_repeats;
        res_avg = res_sum / num_repeats;
        sim_avg = sim_sum / num_repeats;

        % Store the averaged reconstructed flow field and residual
        Reconstructed_Flows(:, t) = x_hat_avg;
        residuals(t) = res_avg;
        sims(t) = sim_avg;

        % Display progress
        disp(['Timestep ', num2str(t), ': Average residual over_____________________________________________________________________ ', num2str(num_repeats), ' repeats = ', num2str(res_avg), '   SSIM = ', num2str(sim_avg)]);
    end
end
