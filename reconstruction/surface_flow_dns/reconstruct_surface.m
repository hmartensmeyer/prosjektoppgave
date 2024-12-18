run("load_data.m");

%%
num_timesteps = 20;
num_repeats = 1;
ns = 20;
sensor_type = 'qr';
SF = 20;
%%
addpath ../../utils/

[recon, res, ssim, sensor_pos] = SPARSE_RECONSTRUCTION(data_normalized(:,1:1050) - mean(data_normalized(:,1:1050), 2), mean(data_normalized(:,1:1050), 2), num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);
[recon_ran, res_ran, ssim_ran, sensor_pos_ran] = SPARSE_RECONSTRUCTION(data_normalized(:,1:1000) - mean(data_normalized(:,1:1050), 2), mean(data_normalized(:,1:1050), 2), num_timesteps, num_repeats, ns, 'random', SF, nx, ny);

%% Display some snapshots of reconstructed flow

% Determine the color limits based on original data
clim = [min(data_normalized(:,:) - mean(mean(data_normalized, 2)))-0.05, max(data_normalized(:,:) - mean(mean(data_normalized, 2)))+0.05];

t = 12;
tl = tiledlayout(1,2,Padding="compact", TileSpacing="compact");

% Reshape the flow fields for plotting
x_true = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon = reshape(recon(:, t), nx, ny); % Adjust nx and ny accordingly

% Plot true flow field
nexttile;
imagesc(x_true);
title(['True Flow Field at Timestep ', num2str(t)]);
colormap (cmap_);
colorbar;
%caxis(clim);

% Plot reconstructed flow field
nexttile;
imagesc(x_recon);
hold on;
title(['Reconstructed Flow Field at Timestep ', num2str(t), ' SSIM = ', num2str(ssim(t))]);
% % Plot sensors, convert linear indices to (row, col) coordinates
% [y_coords, x_coords] = ind2sub([nx, ny], sensor_pos);
% plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
% %plot(sensor_pos(:, 1), sensor_pos(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
colormap (cmap_);
colorbar;
%caxis(clim);

%% Display some snapshots of reconstructed flow

% Determine the color limits based on original data
clim = [min(data_normalized(:)-mean(data_normalized(:), 2))-0.05, max(data_normalized(:)-mean(data_normalized(:), 2))+0.05];

% Choose timesteps to visualize
t = 7;

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2,2, 'TileSpacing', 'tight', 'Padding', 'tight'); % Adjust spacing

% Reshape the flow fields for plotting
x_true = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon_ran = reshape(recon_ran(:, t), nx, ny); % Adjust nx and ny accordingly
x_recon_qr = reshape(recon(:, t), nx, ny); % Adjust nx and ny accordingly


% Plot true flow field
nexttile(1)
imagesc(x_true);
xlabel('Original');
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Plot reconstructed flow field random
nexttile(3)
imagesc(x_recon_ran);
xlabel(['Reconstructed (random)']);
hold on;
plot(sensor_pos_ran(:, 1), sensor_pos_ran(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
hold off;
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Plot reconstructed flow field QR
nexttile(4)
imagesc(x_recon_qr);
hold on;
% Plot sensors, convert linear indices to (row, col) coordinates
[y_coords, x_coords] = ind2sub([nx, ny], sensor_pos);
plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);

xlabel(['Reconstructed (QR)']);
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.8; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;

% Export the figure
%print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%

num_timesteps = 1050;
num_repeats = 10;
ns = 20;
sensor_type = 'random';

% List of undersampling rates
us_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
residuals_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
ssim_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
recon = zeros(n, num_timesteps, length(us_values)); % (n, m, us)
training_set_length = zeros(length(us_values), 1);

for i = 1:length(us_values)
    [Reconstructed_Flow, residuals, ssim] = SPARSE_RECONSTRUCTION(data_normalized(:, 1:1050) - mean(data_normalized(:, 1:1050), 2), mean(data_normalized(:, 1:1050), 2), num_timesteps, num_repeats, ns, sensor_type, us_values(i), nx, ny);

    training_set_length(i) = floor(num_timesteps / us_values(i));
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
save('results_DNS/DNS_20sensors_random_rerun.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length');

save('results_DNS/DNS_20sensors_random_full_rerun.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length', '-v7.3');


%% Define the number of colors in the colormap
n_color = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n_color)', linspace(0, 1, n_color)', ones(n_color, 1); ...
        ones(n_color, 1), linspace(1, 0, n_color)', linspace(1, 0, n_color)'];





%%

disp('hei')





















%% Plot snapshots for "dictionary" cartoon
% mean_flow = mean(data_reshaped,2);
% x = data_reshaped(:, 237) - mean(data_reshaped,2); %random time step
% x_show = data_reshaped(:, 237);
% disp(size(x));
% x_show = reshape(x_show, [nx, ny]);
% 
% figure;
% set(gcf,'Position',[100 100 600 400]);
% imagesc(x_show);
% colorbar;
% colormap winter;
% clim([-0.001, 0.001])
% 
% %% Plot example figure with point measurements
% % Measurement: random pixels
% ns = 100; % Number of point measurements
% 
% % Restrict to cylinder wake: final 80% of width, middle 50% of height
% sensor_idx = [randperm(round(1*ny), ns)' randperm(round(1*nx), ns)']; % Choose sensors on restricted area
% %sensor_idx = [round(1*ny)+sensor_idx(:, 1) round(1*nx)+sensor_idx(:, 2)];  % Translate to wake
% 
% % Convert to measurement matrix
% C = spdiags(ones(n, 1), 0, n, n);
% C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% disp(['C: ', num2str(size(C))]);
% 
% % %Try to make sensing from QR-sensors
% % % Compute the SVD of the snapshot matrix for QR-based measurement selection
% % [U, ~, ~] = svd(data_reshaped(:, 1:5:end), 'econ');
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
% Train = data_reshaped(:, 1:5:end) - mean_flow;
% 
% x = data_reshaped(:, 237) - mean_flow;
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
% [x_hat, res] = reconstruct(x, Train, s, flow, true);
% disp(res)
% 
% %% Plot results
% 
% figure();
% imagesc(reshape(x_hat+mean_flow, nx, ny));
% colorbar;
% set(gcf,'Position',[100 100 600 400]);
% title('xhat');
% colormap winter;
% clim([-0.001, 0.001])

%% Try to reconstruct the full flow field

% Define the flows
Train = data_reshaped(:, 1:10:end);% - mean_flow; %will not subtract mean flow when normalizing

% Measurement: random pixels
% Number of measurements (sensor points)
ns = 30;  % or any desired number of sensors
% % Randomly select sensor indices from the flattened grid
% sensor_linear_idx = randsample(n, ns);
% % Convert linear indices to (row, column) indices
% [sensor_y, sensor_x] = ind2sub([ny, nx], sensor_linear_idx);
% % Combine into a single array for easy access to sensor locations
% sensor_idx = [sensor_y, sensor_x];
% 
% % Convert to measurement matrix
% C = spdiags(ones(n, 1), 0, n, n);
% C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

%Try to make sensing from QR-sensors
% Perform QR decomposition with column pivoting on Train'
[~, ~, E] = qr(Train', 'vector');
E = E';  % Transpose to make it a column vector

% QR-factorization measurements
% Sensor indices selected via QR-factorization
linear_idx_qr = E(1:ns);

% Construct the measurement matrix C_qr
C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);

% Initialize matrix to store reconstructed flow fields
Reconstructed_Flows = zeros(n, 50);
residuals = zeros(50, 1);
sims = zeros(50, 1);
num_repeats = 1;  % Number of times to repeat reconstruction per timestep

for t = 1:50
    % True flow field at timestep t
    x = data_reshaped(:, t);% - mean_flow;

    % Initialize variables to accumulate results over repeats
    x_hat_sum = zeros(n, 1);  % Sum of reconstructed flow fields
    res_sum = 0;              % Sum of residuals
    sim_sum = 0;

    for repeat = 1:num_repeats
        % Sensor measurements
        y = C * x;
        %y = x;
        
        % Measured dictionary
        D = C * Train;
        %D = Train;

        eta=0;
        s = sp_approx(y, D, eta, flow); %original script
        %s = sp_approx(y, D, 1e-7);
        %disp(['s: ', num2str(size(s))]);
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

    % Add mean flow back to the reconstructed flow field
    x_hat_avg_full = x_hat_avg;% + mean_flow;
    
    % Store the averaged reconstructed flow field and residual
    Reconstructed_Flows(:, t) = x_hat_avg_full;
    residuals(t) = res_avg;
    sims(t) = sim_avg;
    
    % Display progress
    disp(['Timestep ', num2str(t), ': Average residual over ', num2str(num_repeats), ' repeats________________________________________________________________ = ', num2str(res_avg), '   SSIM = ', num2str(sim_avg)]);
end

%% Display errors of reconstructed flow

plot(residuals);
hold on
plot(sims);
disp(mean(residuals));

%%

for t = 1:size(Reconstructed_Flows, 2)
    subplot(1,2,1);
    imagesc(reshape(data_reshaped(:, t), nx, ny)); 
    colorbar;
    %caxis([flow_min, flow_max]);
    title(['Original - Timestep ', num2str(t)]);
    
    subplot(1,2,2);
    imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Reconstructed - Timestep ', num2str(t)]);
    colormap winter;
    drawnow; pause(0.001);
end

%%

% Convert linear indices to (row, col) coordinates
[y_coords, x_coords] = ind2sub([nx, ny], linear_idx_qr);

figure;
imagesc(reshape(data_reshaped(:, 1), [nx, ny]));
hold on;
%scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
scatter(x_coords, y_coords, 50, 'k', 'filled'); axis off

%% Display some snapshots of reconstructed flow

% Choose timesteps to visualize
timesteps_to_plot = [16, 35, 102, 150, 200];

for i = 1:length(timesteps_to_plot)
    t = timesteps_to_plot(i);

    % Reshape the flow fields for plotting
    x_true = reshape(data_reshaped(:, t), nx, ny);     % Adjust nx and ny accordingly
    x_recon = reshape(Reconstructed_Flows(:, t), nx, ny); % Adjust nx and ny accordingly

    % Plot true flow field
    figure('Position', [100, 100, 1000, 300]);
    subplot(1,2,1);
    imagesc(x_true);
    title(['True Flow Field at Timestep ', num2str(t)]);
    colormap gray;
    colorbar;
    
    % Plot reconstructed flow field
    subplot(1,2,2);
    imagesc(x_recon);
    title(['Reconstructed Flow Field at Timestep ', num2str(t)]);
    colormap gray;
    colorbar;
end


%% Estimate PSD

addpath '../../utils/'

x_orig = data_reshaped(:, :);
x_rec = Reconstructed_Flows;

disp(size(x_orig))

% Parameters
Nt = size(x_orig, 2);
fs = 1/0.02;  % Sampling frequency (adjust if necessary)
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
us_values = [2, 4, 6, 8, 10, 12,14,16,18,20];
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
    mean_flow = mean(data_reshaped, 2);
    Train = data_reshaped(:, 1:us:end) - mean_flow; % us is the undersampling parameter
    
    % -----------------------------
    % Sensor Selection (QR-based or random)
    % -----------------------------
    
    % Restrict to cylinder wake: final 80% of width, middle 50% of height
    % Randomly select sensor indices from the flattened grid
    sensor_linear_idx = randsample(n, ns);
    % Convert linear indices to (row, column) indices
    [sensor_y, sensor_x] = ind2sub([ny, nx], sensor_linear_idx);
    % Combine into a single array for easy access to sensor locations
    sensor_idx = [sensor_y, sensor_x];
    C = spdiags(ones(n, 1), 0, n, n);
    C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

    % % Compute the SVD of the snapshot matrix
    % addpath '..\..\utils\'
    % 
    % k = 100;
    % [U, ~, ~] = rsvd(Train, k);
    % %U_k = U(:, :); % Use all available modes
    % 
    % % Perform QR decomposition with column pivoting on U_k'
    % [~, ~, E] = qr(U', 'vector');
    % E = E'; % Transpose to make it a column vector
    % 
    % % Sensor indices selected via QR-factorization
    % linear_idx_qr = E(1:ns);
    % 
    % % Construct the measurement matrix C
    % C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
    
    % -----------------------------
    % Sparse Reconstruction
    % -----------------------------
    
    % Initialize matrices to store reconstructed flows and residuals
    Reconstructed_Flows = zeros(n, 60);
    residuals = zeros(60, 1);
    num_repeats = 2; % Number of repeats per timestep (can be adjusted)
    
    for t = 1:30
        % True flow field at timestep t
        x = data_reshaped(:, t) - mean_flow;
        
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
        disp(['us = ', num2str(us), ', Timestep ', num2str(t), ':_____________________________________________________________________________ Residual = ', num2str(res_avg)]);
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
save('output/DNS_US_QR.mat', 'us_values', 'mean_residuals');


%%

disp(['flow.avg_energy: ', num2str(flow.avg_energy)]);
disp(['eps: ', num2str(eps)]);
