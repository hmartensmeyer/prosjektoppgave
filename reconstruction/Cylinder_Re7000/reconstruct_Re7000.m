%% Plot subfigures for diagram of SRC method with cylinder
% Jared Callaham 2018

% Output is partitioned and mean-subtracted Train/Test data and flow params struct
run('load_data.m');

%%
num_timesteps = 10;
num_repeats = 1;
ns = 30;
sensor_type = 'random';
SF = 10;
rescale = true;

%% Convert to grayscale

disp(class(u_field))
disp(diff(getrangefromclass(u_field_reshaped)))

%%
addpath ../../utils/

[recon, res, ssim, sensor] = SPARSE_RECONSTRUCTION_meansub(velocity_magnitude_field_reshaped, flow, num_timesteps, num_repeats, ns, sensor_type, SF, rescale, nx, ny);
[recon_qr, res_qr, ssim_qr, sensor_qr] = SPARSE_RECONSTRUCTION_meansub(velocity_magnitude_field_reshaped, flow, num_timesteps, num_repeats, ns, 'qr', SF, rescale, nx, ny);


%% Display some snapshots of reconstructed flow

% Choose timesteps to visualize
timesteps_to_plot = [2,4,5,6];

for i = 1:length(timesteps_to_plot)
    t = timesteps_to_plot(i);

    % Reshape the flow fields for plotting
    x_true = reshape(velocity_magnitude_field_reshaped(:, t), nx, ny);     % Adjust nx and ny accordingly
    x_recon = reshape(recon(:, t), nx, ny); % Adjust nx and ny accordingly
    x_recon_qr = reshape(recon_qr(:, t), nx, ny);

    % Plot true flow field
    figure('Position', [100, 100, 1700, 400]);
    subplot(1,3,1);
    imagesc(x_true);
    title(['True Flow Field at Timestep ', num2str(t), ' MSE = ', num2str(res(t))]);
    colormap gray;
    colorbar;
    
    % Plot reconstructed flow field
    subplot(1,3,2);
    imagesc(x_recon);
    title(['Random at Timestep ', num2str(t), ' SSIM = ', num2str(ssim(t))]);
    colormap gray;
    colorbar;

    subplot(1,3,3);
    imagesc(x_recon_qr);
    title(['QR at Timestep ', num2str(t), ' SSIM = ', num2str(ssim(t))]);
    colormap gray;
    colorbar;
end

%%

num_timesteps = 600;
num_repeats = 10;
ns = 20;
sensor_type = 'random';
%SF = 10;
rescale = true;

% List of undersampling rates
us_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
residuals_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
ssim_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
recon = zeros(n, num_timesteps, length(us_values)); % (n, m, us)
training_set_length = zeros(length(us_values), 1);

for i = 1:length(us_values)
    [Reconstructed_Flow, residuals, ssim, sensors] = SPARSE_RECONSTRUCTION_meansub(velocity_magnitude_field_reshaped, flow, num_timesteps, num_repeats, ns, sensor_type, us_values(i), rescale, nx, ny);

    training_set_length(i) = floor(m / us_values(i));
    recon(:, :, i) = Reconstructed_Flow; % Store (n, m) for each us
    residuals_(i, :) = residuals;
    ssim_(i, :) = ssim;
end

% Compute mean residuals and SSIM across timesteps for each undersampling rate
mean_residuals = mean(residuals_, 2); % Average over columns (timesteps)
mean_ssim = mean(ssim_, 2);           % Average over columns (timesteps)


% Create the results table
results_table = table(us_values', mean_residuals, mean_ssim, training_set_length, ...
    'VariableNames', {'UndersamplingRate', 'MeanResidual', 'MeanSSIM', 'TrainingSetLength'});

% Display the table
disp(results_table);


% Save all relevant variables to a .mat file
%save('results_Re7000/Re7000_20sensors_random.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length');

%save('results_Re7000/Re7000_20sensors_random_full.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length', '-v7.3');

%%

plot(us_values, mean_ssim);

%%
plot(us_values, mean_residuals);
ylim ([0, 1]);

%%
plot(training_set_length, mean_residuals);


















% %% Plot snapshots for "dictionary" cartoon
% x = u_field_reshaped(:, 237) - mean(u_field_reshaped,2); %random time step
% x_show = u_field_reshaped(:, 237);
% disp(size(x))
% x = reshape(x, [nx, ny]);
% x_show = reshape(x_show, [nx, ny]);
% 
% figure;
% set(gcf,'Position',[100 100 600 400]);
% imagesc(x_show);
% colorbar;
% %clim([-0.2, 0.2])
% 
% %% Plot example figure with point measurements
% % Measurement: random pixels
% ns = 20; % Number of point measurements
% 
% % Restrict to cylinder wake: final 80% of width, middle 50% of height
% sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
% sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
% 
% % Convert to measurement matrix
% C = spdiags(ones(n, 1), 0, n, n);
% C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% disp(['C: ', num2str(size(C))]);
% 
% % %Try to make sensing from QR-sensors
% % % Compute the SVD of the snapshot matrix for QR-based measurement selection
% % [U, ~, ~] = svd(Train, 'econ');
% % %k = 50;  % Number of leading modes to consider
% % U_k = U(:, :);
% % 
% % % Perform QR decomposition with column pivoting on U_k'
% % [~, ~, E] = qr(U_k', 'vector');
% % E = E';  % Transpose to make it a column vector
% % 
% % % QR-factorization measurements
% % % Sensor indices selected via QR-factorization
% % linear_idx_qr = E(1:ns);
% % 
% % % Construct the measurement matrix C_qr
% % C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
% % 
% % % Convert linear indices to (row, col) coordinates
% % [y_coords, x_coords] = ind2sub([nx, ny], linear_idx_qr);
% 
% %%
% figure;
% set(gcf,'Position',[100 100 600 260]);
% imagesc(x_show);
% hold on
% scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
% %scatter(x_coords, y_coords, 50, 'black');
% set(gcf,'Position',[100 100 600 400])
% %clim([-0.2, 0.2])
% 
% 
% 
% %%
% Train = u_field_reshaped(:, 1:10:end);
% 
% x = u_field_reshaped(:, 237);
% 
% D = C*Train;
% %D = C*flow_field;
% 
% % Add noise to the measurement
% eta = 1e-5; %1.65;
% %noise = 0*eta*rms_vort*randn([n, 1]);
% y = C*x;
% 
% % Display the sizes of D and y
% disp(['Size of D (should be [ns, m]): ', num2str(size(D))]);
% disp(['Size of y (should be [ns, 1]): ', num2str(size(y))]);
% 
% % Compute sparse approximation to the entire flow field and calculate error
% s = sp_approx(y, D, eta, flow); %original script
% %s = sp_approx(y, D, 1e-2);
% disp(['s: ', num2str(size(s))]);
% [x_hat, res] = reconstruct(x, Train, s, flow, false);
% disp(res)
% 
% %% Plot results
% 
% figure();
% imagesc(reshape(x_hat, nx, ny));
% colorbar;
% set(gcf,'Position',[100 100 600 400]);
% title('xhat');
% %clim([-0.2, 0.2])

%% Try to reconstruct the full flow field

% Define the flows
mean_flow = (mean(u_field_reshaped, 2));
Train = u_field_reshaped(:, 1:10:end) - mean_flow; % ds is the undersample parameter

% % Testing random sampling of snapshots
% num_snapshots = size(u_field_reshaped, 2);
% % Randomly select 50% of the columns
% random_indices = randperm(num_snapshots, round(0.2 * num_snapshots));
% % Sample the selected snapshots from the training set
% Train = u_field_reshaped(:, random_indices); % Randomly sampled 50% of snapshots

% Measurement: random pixels
ns = 50; % Number of point measurements
% % Restrict to cylinder wake: final 80% of width, middle 50% of height
% sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
% sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
% % Convert to measurement matrix
% C = spdiags(ones(n, 1), 0, n, n);
% C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

%Try to make sensing from QR-sensors
% Compute the SVD of the snapshot matrix for QR-based measurement selection
[U, ~, ~] = svd(Train, 'econ');
k = 30;  % Number of leading modes to consider
U_k = U(:, 1:k);

% Perform QR decomposition with column pivoting on U_k'
[~, ~, E] = qr(Train', 'vector');
E = E';  % Transpose to make it a column vector

% QR-factorization measurements
% Sensor indices selected via QR-factorization
linear_idx_qr = E(1:ns);
% Construct the measurement matrix C_qr
C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);

% Initialize matrix to store reconstructed flow fields
Reconstructed_Flows = zeros(n, 200);
residuals = zeros(200, 1);
num_repeats = 1;  % Number of times to repeat reconstruction per timestep

%%
for t = 1:200
    % True flow field at timestep t
    x = u_field_reshaped(:, t) - mean_flow;

    % Initialize variables to accumulate results over repeats
    x_hat_sum = zeros(n, 1);  % Sum of reconstructed flow fields
    res_sum = 0;              % Sum of residuals
    
    for repeat = 1:num_repeats
        % Sensor measurements
        y = C * x;
        %y = x;
        
        % Measured dictionary
        D = C * Train;
        %D = Train;

        eps=0.09;
        s = sp_approx(y, D, eps, flow); %original script
        %s = sp_approx(y, D, 1e-5);
        %disp(['s: ', num2str(size(s))]);
        [x_hat, res] = reconstruct(x, Train, s, flow, true);
    
        % Accumulate reconstructed flow fields and residuals
        x_hat_sum = x_hat_sum + x_hat;
        res_sum = res_sum + res;
    end

    % Compute average reconstructed flow field and residual for timestep t
    x_hat_avg = x_hat_sum / num_repeats;
    res_avg = res_sum / num_repeats;
    
    % Add mean flow back to the reconstructed flow field
    x_hat_avg_full = x_hat_avg + mean_flow;
    
    % Store the averaged reconstructed flow field and residual
    Reconstructed_Flows(:, t) = x_hat_avg_full;
    residuals(t) = res_avg;
    
    % Display progress
    disp(['Timestep ', num2str(t), ': Average residual over ', num2str(num_repeats), ' repeats = _________________________________________________________', num2str(res_avg)]);
end

%% Display errors of reconstructed flow

plot(residuals);
disp(mean(residuals));

%% Display some snapshots of reconstructed flow

% Choose timesteps to visualize
timesteps_to_plot = [16, 35, 102, 150, 200];

for i = 1:length(timesteps_to_plot)
    t = timesteps_to_plot(i);

    % Reshape the flow fields for plotting
    x_true = reshape(u_field_reshaped(:, t), nx, ny);     % Adjust nx and ny accordingly
    x_recon = reshape(Reconstructed_Flows(:, t), nx, ny); % Adjust nx and ny accordingly

    % Plot true flow field
    figure('Position', [100, 100, 1000, 300]);
    subplot(1,2,1);
    imagesc(x_true);
    title(['True Flow Field at Timestep ', num2str(t)]);
    %colormap jet;
    colorbar;
    
    % Plot reconstructed flow field
    subplot(1,2,2);
    imagesc(x_recon);
    title(['Reconstructed Flow Field at Timestep ', num2str(t)]);
    %colormap jet;
    colorbar;
end

%%

% Convert linear indices to (row, col) coordinates
[y_coords, x_coords] = ind2sub([nx, ny], linear_idx_qr);

figure;
imagesc(reshape(u_field_reshaped(:, 1), [nx, ny]));
hold on;
%scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
scatter(x_coords, y_coords, 50, 'k', 'filled'); axis off

%%

for t = 1:size(Reconstructed_Flows, 2)
    subplot(1,2,1);
    imagesc(reshape(u_field_reshaped(:, t), nx, ny));
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Original - Timestep ', num2str(t)]);
    
    subplot(1,2,2);
    imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Reconstructed - Timestep ', num2str(t)]);
    
    drawnow; pause(0.1);
end

%%

for t = 1:size(Reconstructed_Flows, 2)
    subplot(1,2,1);
    imagesc(reshape(u_field_reshaped(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Original - Timestep ', num2str(t)]);
    
    subplot(1,2,2);
    imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Reconstructed - Timestep ', num2str(t)]);
    
    drawnow; pause(0.1);
end

%% Estimate PSD

addpath '../../utils/'

x_orig = u_field_reshaped(:, :);
x_rec = Reconstructed_Flows;

disp(size(x_rec))

% Parameters
Nt = size(x_orig, 2);
fs = 15;  % Sampling frequency (adjust if necessary)
N = 2^nextpow2(Nt);  % Ensure N is a power of 2 for FFT

% Calculate PSD for X, L, and S using the estpsd function
[Gdat_orig, f_orig, e_r_orig] = estpsd(x_orig, N, fs);
[Gdat_rec, f_rec, e_r_rec] = estpsd(x_rec, N, fs);

% Average PSD across all spatial points (channels)
PSD_orig = mean(Gdat_orig, 2);
PSD_rec = mean(Gdat_rec, 2);

% Plot the PSDs with publication-quality settings
figure('Units', 'inches', 'Position', [0, 0, 6, 4]);

% Plot each dataset with distinct line styles and markers
loglog(f_orig, PSD_orig, 'b', 'LineWidth', 1.5, 'DisplayName', 'Original');
hold on;
loglog(f_rec, PSD_rec, 'k', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed');

% Customize labels and legend
xlabel('Frequency (Hz)', 'FontSize', 14);
ylabel('Power Spectral Density', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 12, 'Box', 'off');

% Customize grid and axes
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', ...
         'XMinorTick', 'on', 'YMinorTick', 'on', 'TickLength', [0.02, 0.02]);

% Adjust axes limits if necessary
axis tight;
hold off;

% Save the figure in a high-resolution vector format
% Define the output directory
output_dir = 'output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Adjust the paper size to match the figure size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, 6, 4]);
set(gcf, 'PaperSize', [6, 4]);

%%

% List of undersampling rates
us_values = [2, 5, 10, 20, 30];
mean_residuals = zeros(length(us_values), 1); % To store mean residuals for each us

% Number of point measurements
ns = 30;

% Loop over each undersampling rate
for idx = 1:length(us_values)
    us = us_values(idx);
    
    % -----------------------------
    % Data Preparation
    % -----------------------------
    
    % Define the mean flow and training data
    mean_flow = (mean(u_field_reshaped, 2));
    Train = u_field_reshaped(:, 1:us:end) - mean_flow; % us is the undersampling parameter
    
    % -----------------------------
    % Sensor Selection (QR-based or random)
    % -----------------------------
    
    % % Restrict to cylinder wake: final 80% of width, middle 50% of height
    % sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
    % sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
    % % Convert to measurement matrix
    % C = spdiags(ones(n, 1), 0, n, n);
    % C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

    % Compute the SVD of the snapshot matrix
    [U, ~, ~] = svd(Train, 'econ');
    U_k = U(:, 1:ns); % Use all available modes
    
    % Perform QR decomposition with column pivoting on U_k'
    [~, ~, E] = qr(U_k', 'vector');
    E = E'; % Transpose to make it a column vector
    
    % Sensor indices selected via QR-factorization
    linear_idx_qr = E(1:ns);
    
    % Construct the measurement matrix C
    C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
    
    % -----------------------------
    % Sparse Reconstruction
    % -----------------------------
    
    % Initialize matrices to store reconstructed flows and residuals
    Reconstructed_Flows = zeros(n, 60);
    residuals = zeros(60, 1);
    num_repeats = 1; % Number of repeats per timestep (can be adjusted)
    
    for t = 1:60
        % True flow field at timestep t
        x = u_field_reshaped(:, t) - mean_flow;
        
        % Initialize variables to accumulate results over repeats
        x_hat_sum = zeros(n, 1); % Sum of reconstructed flow fields
        res_sum = 0;             % Sum of residuals
        
        for repeat = 1:num_repeats
            % Sensor measurements
            y = C * x;
            
            % Measured dictionary
            D = C * Train;
            
            % Sparse approximation
            eta = 0;
            s = sp_approx(y, D, eta, flow); % Modify as needed based on your sp_approx function
            %s = sp_approx(y, D, 1e-5); %for LASSO
            
            % Reconstruction
            [x_hat, res] = reconstruct(x, Train, s, flow, true); % Modify as needed based on your reconstruct function
            
            % Accumulate results
            x_hat_sum = x_hat_sum + x_hat;
            res_sum = res_sum + res;
        end
        
        % Compute average reconstructed flow field and residual
        x_hat_avg = x_hat_sum / num_repeats;
        res_avg = res_sum / num_repeats;
        
        % Add mean flow back to the reconstructed flow field
        x_hat_avg_full = x_hat_avg + mean_flow;
        
        % Store the reconstructed flow and residual
        Reconstructed_Flows(:, t) = x_hat_avg_full;
        residuals(t) = res_avg;
        
        % Display progress
        disp(['us = ', num2str(us), ', Timestep ', num2str(t), ':                                                                               Residual = ', num2str(res_avg)]);
    end
    
    % Compute and store the mean residual for this undersampling rate
    valid_residuals = residuals(~isnan(residuals));
    if ~isempty(valid_residuals)
        mean_residual = mean(valid_residuals);
    else
        mean_residual = NaN;
    end
    mean_residuals(idx) = mean_residual;
    disp(['us = ', num2str(us), ', Mean residual over all timesteps: ', num2str(mean_residual)]);
end


% Save the array of mean residuals as a function of undersampling rates
results_table = table(us_values', mean_residuals, 'VariableNames', {'UndersamplingRate', 'MeanResidual'});
disp(results_table);

%%

train_length = round([200, 100, 200/5, 200/10, 10, 200/30]);

% Plot the mean residuals versus undersampling rates
figure;
plot(us_values, mean_residuals, '-o', 'LineWidth', 2);
xlabel('Undersampling Rate (us)');
ylabel('Mean Residual');
grid on;

% figure;
% plot(train_length, mean_residuals, '-o', 'LineWidth', 2);
% xlabel('Training set length');
% ylabel('Mean Residual');
% grid on;


%%

% Save the undersampling rates and corresponding mean residuals to a MAT-file
save('output/Re7000_US_QR.mat', 'us_values', 'mean_residuals');


%%

disp(['flow.avg_energy: ', num2str(flow.avg_energy)]);
disp(['eps: ', num2str(eps)]);


%%


% Define the flows
mean_flow = mean(u_field_reshaped, 2);
Train = u_field_reshaped(:, 1:3:end) - mean_flow;

% Compute the SVD of the snapshot matrix
[U, ~, ~] = svd(Train, 'econ');
k = 20;  % Number of leading modes to retain
U_k = U(:, 1:k);

% Perform QR decomposition with column pivoting on U_k'
[~, ~, E] = qr(U_k', 'vector');
linear_idx_qr = E(1:ns);  % Sensor indices selected via QDEIM

% Number of sensors
ns = 20;

% Initialize matrix to store reconstructed flow fields
Reconstructed_Flows = zeros(n, 60);
residuals = zeros(60, 1);

for t = 1:60
    % True flow field at timestep t
    x = u_field_reshaped(:, t);
    x_mean_subtracted = x - mean_flow;

    % Measurements at interpolation points
    y = x_mean_subtracted(linear_idx_qr);

    % Extract U_k at interpolation points
    U_k_I = U_k(linear_idx_qr, :);

    % Solve for alpha (coefficients)
    alpha = U_k_I \ y;  % Least squares solution

    % Reconstruct the flow field
    x_hat = U_k * alpha + mean_flow;

    % Compute residual
    res = norm(x - x_hat) / norm(x);

    % Store results
    Reconstructed_Flows(:, t) = x_hat;
    residuals(t) = res;

    % Display progress
    disp(['Timestep ', num2str(t), ': Residual = ', num2str(res)]);
end

%%

for t = 1:size(Reconstructed_Flows, 2)
    subplot(1,2,1);
    imagesc(reshape(u_field_reshaped(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Original - Timestep ', num2str(t)]);
    
    subplot(1,2,2);
    imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Reconstructed - Timestep ', num2str(t)]);
    
    drawnow; pause(0.1);
end

