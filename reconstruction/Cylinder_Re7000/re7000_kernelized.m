%% Plot subfigures for diagram of SRC method with cylinder
% Jared Callaham 2018

% Output is partitioned and mean-subtracted Train/Test data and flow params struct
run('load_cylinder_re7000.m');

%% Plot snapshot
x = u_field_reshaped(:, 153) - mean(u_field_reshaped,2); %random time step
x_show = u_field_reshaped(:, 153);
disp(size(x))
x = reshape(x, [nx, ny]);
x_show = reshape(x_show, [nx, ny]);

figure;
set(gcf,'Position',[100 100 600 400]);
imagesc(x_show);
colorbar;

%% Construct linearly growing windows as a function of downstream coordinate

% Automatically build linearly growing windows
window_width = @(x) floor(0.5*(x/ny)*nx) + 20;
%window_width = 30;
x0 = 1;
window_end = x0 + window_width(x0);  % Keep track of where the last window is
while window_end < ny  % Note that last one will be too far (stop the reconstruction loop short)
    x0 = [x0 window_end]; 
    window_end = window_end + window_width(window_end);
    disp(window_end)
end
% Extend last window to end of domain
x0(end) = nx;
k = length(x0)-1;  % Number of kernels/windows
ns = 20;  % Evenly spaced along midline


%% Define and Normalize Kernels

% Define kernel centers (midpoints of windows)
kernel_x = floor(x0(1:end-1) + 0.5 * diff(x0));  % Center x-coordinates of kernels
kernel_y = ny / 2 + zeros(size(kernel_x));       % Center y-coordinate (midline)

% Define kernel width (adjust as needed)
kernel_width = nx / (2 * k);  % Controls the spread of the kernel
phi = @(r) exp(-r.^2 / kernel_width^2);  % Gaussian kernel function

% Initialize variables to store kernels and normalization denominator
Phi = cell(k, 1);  % Cell array to store kernels
denom = zeros(ny, nx);  % Denominator for normalization

% Compute kernels and accumulate for normalization
for j = 1:k
    [X_grid, Y_grid] = meshgrid(1:nx, 1:ny);
    XX = X_grid - kernel_x(j);
    YY = Y_grid - kernel_y(j);
    RR = sqrt(XX.^2 + YY.^2);  % Distance from kernel center
    Phi{j} = phi(RR);
    denom = denom + Phi{j};  % Accumulate for normalization
end

% Normalize kernels and convert to sparse diagonal matrices
for j = 1:k
    kernel = Phi{j} ./ denom;     % Normalize kernel
    kernel(kernel < 1e-2) = 0;    % Threshold small values to zero
    Phi{j} = spdiags(kernel(:), 0, n, n);  % Convert to sparse diagonal matrix
end

%%

% Number of windows (regions)
k = length(x0) - 1;

% Sensor configurations per region
ns_list = [5, 10, 20, 30];           % Number of sensors per region
percentage_list = [0.4, 0.5, 0.7, 1]; % Vertical coverage per region (as a fraction of ny)

% Initialize cell array to store sensor positions
sensor_positions = cell(k, 1);

% Initialize vector to collect all sensor linear indices
sensor_linear_indices = [];

for i = 1:k
    % Region boundaries in x-direction
    x_start = x0(i);
    x_end = x0(i+1);
    
    % Number of sensors and vertical coverage for this region
    ns = ns_list(i);
    percentage = percentage_list(i);
    
    % Define vertical range centered around the middle of the domain
    y_center = nx / 2;
    y_extent = (percentage * nx) / 2;
    y_start = round(y_center - y_extent);
    y_end = round(y_center + y_extent);
    y_start = max(y_start, 1);
    y_end = min(y_end, ny);
    
    % Create grid of possible sensor locations within the region
    [X_grid, Y_grid] = meshgrid(x_start:x_end, y_start:y_end);
    X_grid = X_grid(:);
    Y_grid = Y_grid(:);
    
    % Total possible sensor locations in this region
    possible_indices = [X_grid, Y_grid];
    num_possible = size(possible_indices, 1);
    
    % Randomly select sensor positions
    ns_actual = min(ns, num_possible);
    rand_idx = randperm(num_possible, ns_actual);
    selected_positions = possible_indices(rand_idx, :); % [x, y] positions
    sensor_positions{i} = selected_positions;
    
    % Compute linear indices for sensors in the full flow field
    % Linear index in MATLAB: linear_idx = row + (col - 1) * number_of_rows
    sensor_full_indices = selected_positions(:,1) + (selected_positions(:,2) - 1) * nx;
    
    % Collect all sensor linear indices
    sensor_linear_indices = [sensor_linear_indices; sensor_full_indices];
end

% Total number of grid points
n = nx * ny;

% Total number of sensors
ns_total = length(sensor_linear_indices);

% Construct measurement matrix C
C = sparse(1:ns_total, sensor_linear_indices, ones(ns_total,1), ns_total, n);


%% Plot snapshot with windows
figure;
set(gcf,'Position',[100 100 600 400]);
imagesc(x_show);
colorbar;
hold on;

% Plot windows as lines on the snapshot
for i = 1:k
    x_start = x0(i);
    x_end = x0(i + 1);
    plot([x_start, x_start], [1, nx], 'r--', 'LineWidth', 2); % Starting line of window
    plot([x_end, x_end], [1, nx], 'r--', 'LineWidth', 2); % Ending line of window
end

xlabel('x-coordinate');
ylabel('y-coordinate');
title('Flow Field with Linearly Growing Windows');
hold off;

%%


figure;
set(gcf,'Position',[100 100 600 400]);
imagesc(x_show);
set(gca, 'YDir', 'normal'); % Ensure y-axis increases upwards
colorbar;
hold on;

% Plot windows as lines on the snapshot
for i = 1:k
    x_start = x0(i);
    x_end = x0(i + 1);
    plot([x_start, x_start], [1, nx], 'r--', 'LineWidth', 2); % Starting line of window
    plot([x_end, x_end], [1, nx], 'r--', 'LineWidth', 2); % Ending line of window
end

% Plot sensor positions
for i = 1:k
    sensors = sensor_positions{i};
    plot(sensors(:,1), sensors(:,2), 'ko', 'MarkerFaceColor', 'black', 'MarkerSize', 6);
end

xlabel('x-coordinate');
ylabel('y-coordinate');
title('Flow Field with Linearly Growing Windows and Sensor Positions');
hold off;

%%

% Number of total grid points
n = nx * ny;

% Initialize the reconstructed flow field and a weight matrix
x_hat = zeros(n, 1);  % Reconstructed flow field
w = zeros(n, 1);      % Weight matrix to handle overlapping regions

% Define training samples (adjust the step as needed)
mean_flow = mean(mean(u_field_reshaped, 2));
training_samples = u_field_reshaped(:, 1:5:end) - mean_flow;

% Original snapshot to reconstruct
t = 153;

%%

% --- Loop Over Each Region for Reconstruction ---

for i = 1:k  % Loop over each window/region

    % Define x_start and x_end for the current region
    x_start = x0(i);
    x_end = x0(i+1);
    
    % Define x and y ranges for the current region
    x_range = x_start:x_end;     % x-coordinate (columns)
    y_range = 1:ny;              % y-coordinate (rows)
    
    % Create grid of points
    [X_grid, Y_grid] = meshgrid(x_range, y_range);
    
    % Linear indices for the region
    region_indices = Y_grid + (X_grid - 1) * ny;
    region_indices = region_indices(:);
    
    % Extract the kernel for this region
    kernel = Phi{i};
    
    % Apply kernel to the snapshot to reconstruct
    x_kernel = kernel * u_field_reshaped(:, t);
    
    % Apply kernel to the training data
    Train_i = kernel * (u_field_reshaped(:, 1:2:end) - mean_flow);
    
    % Extract the measurement matrix for the region (no change needed)
    C_i = C;  % Since C is global and corresponds to sensor measurements
    
    % Measurement vector for the region
    y_i = C_i * x_kernel;
    
    % Dictionary for the sparse approximation
    D_i = C_i * Train_i;
    
    % Perform sparse approximation (adjust 'eta' and 'flow' as needed)
    s_i = sp_approx(y_i, D_i, eta, flow);
    
    % Reconstruct the flow field in the region
    [x_hat_i, res] = reconstruct(x_kernel, Train_i, s_i, flow, false);
    
    % Update the reconstructed flow field and weight matrix
    x_hat = x_hat + x_hat_i;
    w = w + 1;
    
    disp(['Finished with step: ', num2str(i)])
end


%%

% Compute reconstruction error
reconstruction_error = norm(x_hat - u_field_reshaped(:, 400)) / norm(u_field_reshaped(:, 400) + mean_flow);
disp(['Reconstruction Error: ', num2str(reconstruction_error)]);


%%

disp(size(x_hat))
x_hat_reshaped = reshape(x_hat, [nx, ny]);
x_hat_reshaped = x_hat_reshaped + mean_flow;
disp(size(x_hat_reshaped))
figure;
imagesc(x_hat_reshaped);
colorbar;

%%