run("load_data.m");

%%
num_timesteps = 20;
num_repeats = 1;
ns = 13;
sensor_type = 'cluster';
SF = 10;

% %% SVD of data_reshaped and then throw away first mode
% 
% addpath ..\..\utils\
% 
% % Perform SVD on the reshaped datam, already mean-subtracted in load_data
% [U, Sigma, V] = svd(data_reshaped, 'econ');
% 
% % %% Gavish & Donoho optimal threshold (nothing good here)
% % %[U_ot D_ot V_ot] = svd(data_reshaped - mean(data_reshaped), 'econ');
% % 
% % y = diag(Sigma);
% % y( y < (optimal_SVHT_coef(m/n,0) * sqrt(n) * median(y)) ) = 0;
% % Xhat = U * diag(y) * V';
% % 
% % disp(size(y));
% % 
% % %%
% % y = diag(Sigma);
% % cutoff = (optimal_SVHT_coef(m/n,0) * sqrt(n) * median(y));
% % 
% % %%
% % 
% % disp(max(y));
% % disp(cutoff);
% % 
% % %%
% % 
% % loglog(y, 'o-');
% % hold on;
% % plot([1, 1000], [cutoff, cutoff])
% 
% %% Choose the number of modes to plot and reconstruct
% num_modes = [1, 2, 3, 4, 5]; % Modes to visualize
% 
% hfig = figure;
% 
% % Create a tiled layout for better visualization
% num_plots = length(num_modes); % Number of modes to plot
% t = tiledlayout(1, num_plots, 'TileSpacing', 'compact', 'Padding', 'tight'); % One row
% 
% % Plot the selected modes
% for k = 1:num_plots
%     nexttile; % Move to the next tile
%     mode = num_modes(k);
%     imagesc(reshape(U(:, mode), nx, ny)); % Reshape to spatial dimensions
%     colormap gray;
%     title(['Mode ', num2str(mode)]);
%     set(gca, 'XTick', [], 'YTick', [])
% end
% 
% fname = 'output_DNS/PSD_DNS_lambda_tuning';
% 
% picturewidth = 20; % set this parameter and keep it forever
% hw_ratio = 0.3; % feel free to play with this ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
% legend('Location', 'southwest', 'FontSize', 21, 'FontWeight','bold');
% set(findall(hfig,'-property','Box'),'Box','off') % optional
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
% box on;
% 
% %Saver
% %print(hfig,fname,'-dpdf','-vector');
% %print(hfig,fname,'-dpng','-vector')
% 
% %%
% 
% starter = 2;
% ender = 809;
% 
% % Reconstruct the data using the first few modes
% data_reconstructed = U(:, starter:ender) * Sigma(starter:ender, starter:ender) * V(:, starter:ender)';
% 
% % Verify reconstruction quality by plotting the first frame
% figure;
% imagesc(reshape(data_reconstructed(:, 100), nx, ny));
% colormap gray;
% colorbar;
% title(['Reconstructed Frame 1 using ', num2str(ender-starter), ' modes']);
% 
% %%
% for t = 1:1000
%     imagesc(reshape(data_reconstructed(:, t), nx, ny));
%     colorbar;
%     %clim([0, 50])
%     pause(0.02)
%     colormap gray;
% end
% 
% %%
% % Extract singular values from the diagonal of Sigma
% singular_values = diag(Sigma);
% 
% % Plot the singular values
% figure;
% loglog(singular_values / sum(singular_values), 'o-', 'LineWidth', 1.5);
% xlabel('Index');
% ylabel('Singular Value');
% title('Singular Values of data\_reshaped');
% grid on;
% 
% figure;
% semilogx(cumsum(singular_values) / sum(singular_values), 'o-');
% grid on;
% 
% %% Normalization of SVD-recon
% 
% % Min-Max Normalization per column
% min_vals = min(data_reconstructed, [], 1);
% max_vals = max(data_reconstructed, [], 1);
% epsilon = 1e-8; % To avoid division by zero
% data_reconstructed = (data_reconstructed - min_vals) ./ (max_vals - min_vals + epsilon);
% 
% %% Testing to do RPCA before reconstruction
% 
% [L_1, S_1] = ALM_RPCA(data_normalized(:, 1:100), 1, 1e-5, 1000);
% %%
% [L_03, S_03] = ALM_RPCA(data_normalized(:, 1:300), 0.6, 1e-5, 1000);
% %%
% t=21;
% 
% hfig = figure('Position', [100,100,600,600])
% t_ = tiledlayout(2,2, Padding='tight', TileSpacing='compact')
% 
% nexttile;
% imagesc(reshape(data_normalized(:, t), nx, ny))
% colormap gray;
% 
% nexttile;
% imagesc(reshape(recon_og_clu(:, t), nx, ny))
% title(res_og_clu(t), ssim_og_clu(t))
% colormap gray;
% nexttile;
% imagesc(reshape(recon_og_r(:, t), nx, ny))
% title(res_og_r(t), ssim_og_r(t))
% colormap gray;
% 
% %%
% disp('---SSIM---')
% disp(ssim_og_clu(t))
% disp(ssim_og_r(t))
% disp('---MSE---')
% disp(res_og_clu(t))
% disp(res_og_r(t))
% disp('---PSNR---')
% disp(psnr(data_normalized(:, t), recon_og_clu(:, t)))
% disp(psnr(data_normalized(:, t), recon_og_r(:, t)))
% 
% %% Normalization of S
% 
% % Min-Max Normalization per column
% min_vals = min(S_1, [], 1);
% max_vals = max(S_1, [], 1);
% epsilon = 1e-8; % To avoid division by zero
% S_1 = (S_1 - min_vals) ./ (max_vals - min_vals + epsilon);
% 
% % Min-Max Normalization per column
% min_vals = min(S_03, [], 1);
% max_vals = max(S_03, [], 1);
% epsilon = 1e-8; % To avoid division by zero
% S_03 = (S_03 - min_vals) ./ (max_vals - min_vals + epsilon);

%% SR on the raw data
addpath ../../utils/

%[recon_og_clu, res_og_clu, ssim_og_clu, sensor_pos_og_clu] = SPARSE_RECONSTRUCTION(data_normalized - mean(data_normalized, 2),  mean(data_normalized, 2), num_timesteps, num_repeats, ns, 'QR_cluster', SF, nx, ny);
[recon_og_r, res_og_r, ssim_og_r] = SPARSE_RECONSTRUCTION(data_normalized,  mean(data_normalized, 2), num_timesteps, num_repeats, ns, 'random_cluster', SF, nx, ny);

% %% SR on the SVD data, the the first mode (color gradient) and the last modes are dropped
% 
% [recon_svd_clu, res_svd_clu, ssim_svd_clu, sensor_pos_svd_clu] = SPARSE_RECONSTRUCTION(data_reconstructed, flow, num_timesteps, num_repeats, ns, 'random_cluster', SF, nx, ny);
% [recon_svd_r, res_svd_r, ssim_svd_r, sensor_pos_svd_r] = SPARSE_RECONSTRUCTION(data_reconstructed, flow, num_timesteps, num_repeats, ns, 'random', SF, nx, ny);
% 
% %% SR on RPCA data
% [recon_rpca_1, res_rpca_1, ssim_rpca_1, sensor_pos_rpca_1] = SPARSE_RECONSTRUCTION(S_1, flow, num_timesteps, num_repeats, ns, 'random_cluster', SF, nx, ny);
% [recon_rpca_03, res_rpca_03, ssim_rpca_03, sensor_pos_rpca_03] = SPARSE_RECONSTRUCTION(S_03, flow, num_timesteps, num_repeats, ns, 'random_cluster', SF, nx, ny);
% 
% %% display some snapshots
% 
%figure('Position', [10, 100, 1600, 400]);

%%

t = 15;
hfig = figure;
tl = tiledlayout(1,2, Padding="compact",TileSpacing="compact");

nexttile;
imagesc(reshape(data_normalized(:, t) - mean(data_normalized, 2), nx, ny));
%title(t)
%colorbar;
xlabel('Original')
colormap gray;
%axis off;
set(gca, 'XTick', [], 'YTick', []);

nexttile;
imagesc(reshape(recon_og_r(:, t) - mean(data_normalized, 2), nx, ny));
title(ssim_og_r(t), res_og_r(t))
hold on;
colormap gray;
xlabel('Reconstructed');
%axis off;
set(gca, 'XTick', [], 'YTick', []);

fname = 'output_Nidelva/SR_comparison_ pixelsensors';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.45; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
%legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','on') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);

%print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

%%

plot(res_og_clu);
disp(mean(res_og_clu))
%plot(ssim_og_clu);
disp(mean(ssim_og_clu))

%% Plot illustative picture:

t = 160;
hfig = figure;

imagesc(reshape(data_normalized(:, t), nx, ny));
%title(t)
%colorbar;
%xlabel('Original')
colormap gray;
%axis off;
set(gca, 'XTick', [], 'YTick', []);

fname = 'output_theory/illustration_river_flow';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.8; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
%legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','on') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);

print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

% 
% %%
% 
% % Create a binary map where 1 indicates sensor coverage
% sensor_map = any(sensor_pos_svd_clu, 1); % 1 x 291600 logical vector
% 
% % Reshape to 2D grid
% sensor_image = reshape(sensor_map, ny, nx);
% 
% figure;
% imagesc(sensor_image);
% colormap(gray);
% title('Sensor Coverage');
% axis image; % Maintain aspect ratio
% colorbar;

%% plot errors

plot(ssim_og_clu, 'DisplayName','raw (cluster)', 'LineWidth',2, 'LineStyle','-.')
hold on;
plot(ssim_og_r, 'DisplayName','raw (pixel)', 'LineWidth',2, 'LineStyle','-')
%plot(ssim_svd_clu, 'DisplayName', 'svd (cluster)', 'LineWidth',2, 'LineStyle',':')
%plot(ssim_svd_r, 'DisplayName', 'svd (pixel)', 'LineWidth',2, 'LineStyle','--')
%plot(ssim_rpca_1, 'DisplayName', 'rpca (kappa=1=', 'LineWidth',2, 'LineStyle','--')
%plot(ssim_rpca_03, 'DisplayName', 'rpca - (kappa=0.6)', 'LineWidth',2, 'LineStyle','--')
legend;

%%
plot(res_og_clu, 'DisplayName', 'raw (cluster)', 'LineWidth', 2, 'LineStyle', '-.');
hold on;
plot(res_og_r, 'DisplayName', 'raw (pixel)', 'LineWidth', 2, 'LineStyle', '-');
%plot(res_svd_clu, 'DisplayName', 'svd (cluster)', 'LineWidth', 2, 'LineStyle', ':');
%plot(res_svd_r, 'DisplayName', 'svd (pixel)', 'LineWidth', 2, 'LineStyle', '--');
%plot(res_rpca_1, 'DisplayName', 'rpca (kappa=1)', 'LineWidth', 2, 'LineStyle', '--');
%plot(res_rpca_03, 'DisplayName', 'rpca (kappa=0.6)', 'LineWidth', 2, 'LineStyle', '--');
legend;


%%

fprintf('Raw, cluster: %.4f\n', mean(res_og_clu));
fprintf('Raw, pixels: %.4f\n', mean(res_og_r));
%fprintf('SVD, cluster: %.4f\n', mean(res_svd_clu));
%fprintf('SVD, pixels: %.4f\n', mean(res_svd_r));
%fprintf('RPCA (1), cluster: %.4f\n', mean(res_rpca_1));
%fprintf('RPCA (0.3), cluster: %.4f\n', mean(res_rpca_03));

%%

fprintf('Raw, cluster: %.4f\n', mean(ssim_og_clu));
fprintf('Raw, pixels: %.4f\n', mean(ssim_og_r));
%fprintf('SVD, cluster: %.4f\n', mean(ssim_svd_clu));
%fprintf('SVD, pixels: %.4f\n', mean(ssim_svd_r));
%fprintf('RPCA (1), cluster: %.4f\n', mean(ssim_rpca_1));
%fprintf('RPCA (0.3), cluster: %.4f\n', mean(ssim_rpca_03));

%% Loop over no. of sensors

% Parameters
num_timesteps = 1050;
num_repeats = 5;
sensor_type = 'random_cluster'; 
SF = 5;
max_ns = 31;

% Preallocate storage
recon_og_clu = cell(max_ns,1);       % Reconstructions
res_og_clu = zeros(max_ns, num_timesteps);         % Residuals
ssim_og_clu = zeros(max_ns, num_timesteps); % SSIM values
sensor_pos_og_clu = cell(max_ns,1);  % Sensor positions
mse_og_clu = zeros(max_ns,1);        % MSE values
measurements = zeros(max_ns,1);       % Number of measured grid points

% Loop over number of sensors from 1 to 50
for ns = 1:2:max_ns
    % Perform Sparse Reconstruction
    [recon, res, ssim, sensor_pos] = SPARSE_RECONSTRUCTION(...
        data_normalized - mean(data_normalized, 2), mean(data_normalized, 2), num_timesteps, num_repeats, ns, ...
        sensor_type, SF, nx, ny); 
    
    % Store results
    recon_og_clu{ns} = recon;
    res_og_clu(ns, :) = res;
    ssim_og_clu(ns, :) = ssim;
    sensor_pos_og_clu{ns} = sensor_pos;
    
    % Compute Number of Measured Grid Points
    num_measurements = ns * 20 * 20;
    measurements(ns) = num_measurements;
    
    % Display Progress
    fprintf('Completed reconstruction with %d sensors.\n', ns);
end

% Save all relevant variables to a .mat file
save('results_Nidelva/error_sensor_loop_US5_meansub.mat', 'recon_og_clu', 'res_og_clu', 'ssim_og_clu', '-v7.3');

%% Loop over no. of sensors

% Parameters
num_timesteps = 1050;
num_repeats = 5;
sensor_type = 'random_cluster'; 
SF = 10;
max_ns = 31;

% Preallocate storage
recon_og_clu = cell(max_ns,1);       % Reconstructions
res_og_clu = zeros(max_ns, num_timesteps);         % Residuals
ssim_og_clu = zeros(max_ns, num_timesteps); % SSIM values
sensor_pos_og_clu = cell(max_ns,1);  % Sensor positions
mse_og_clu = zeros(max_ns,1);        % MSE values
measurements = zeros(max_ns,1);       % Number of measured grid points

% Loop over number of sensors from 1 to 50
for ns = 1:2:max_ns
    % Perform Sparse Reconstruction
    [recon, res, ssim, sensor_pos] = SPARSE_RECONSTRUCTION(...
        data_normalized - mean(data_normalized, 2), mean(data_normalized, 2), num_timesteps, num_repeats, ns, ...
        sensor_type, SF, nx, ny); 
    
    % Store results
    recon_og_clu{ns} = recon;
    res_og_clu(ns, :) = res;
    ssim_og_clu(ns, :) = ssim;
    sensor_pos_og_clu{ns} = sensor_pos;
    
    % Compute Number of Measured Grid Points
    num_measurements = ns * 20 * 20;
    measurements(ns) = num_measurements;
    
    % Display Progress
    fprintf('Completed reconstruction with %d sensors.\n', ns);
end

% Save all relevant variables to a .mat file
save('results_Nidelva/error_sensor_loop_US10_meansub.mat', 'recon_og_clu', 'res_og_clu', 'ssim_og_clu', '-v7.3');

%% Example: Plot SSIM vs Number of Sensors
hfig = figure;
mean_ssim = mean(ssim_og_clu, 2); % Average SSIM over timesteps
plot(1:2:31, mean_ssim(1:2:end), 'ko-', 'LineWidth',1.5);
xlabel('Sensors');
ylabel('SSIM');
xlim([1,32])
grid on;

% Set additional properties
fname = 'output_Nidelva/ssim_us5_sensortuning';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

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

%% Example: Plot MSE vs Number of Sensors

hfig = figure;
mean_res = mean(res_og_clu, 2);
plot(1:2:max_ns, mean_res(1:2:31), 'k-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Sensors');
ylabel('Mean square error');
grid on;
xlim([1 max_ns]);

% Set additional properties
fname = 'output_Nidelva/mse_us10_sensortuning';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

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

% %% Plot MSE vs measured grid points
% 
% tester = 1:50;
% measurements = 20*20*tester;
% figure;
% plot(0.1 + (measurements / (540*540)), mean_res, '-o', 'LineWidth', 2);
% xlabel('Ratio of Measured Grid Points');
% ylabel('Mean Squared Error (MSE)');
% title('MSE vs. Number of Measured Grid Points');
% grid on;

%%

hfig = figure;
tiledlayout(2, 2, "TileSpacing","compact",'Padding','compact');
timestep = 889; %% 886 for figs, 673 shows 11 underperforming

nexttile;
imagesc(reshape(data_normalized(:, timestep), nx, ny))
colormap gray;
xlabel('Original')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_og_clu{3}(:, timestep), nx, ny))
colormap gray;
xlabel('Reconstructed, 3 sensors')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_og_clu{11}(:, timestep), nx, ny))
colormap gray;
xlabel('Reconstructed, 11 sensors')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_og_clu{31}(:, timestep), nx, ny))
colormap gray;
xlabel('Reconstructed, 31 sensors')
set(gca, 'XTick', [], 'YTick', [])

% Set additional properties
fname = 'output_Nidelva/og_vs_recon_3_13_31';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.9; % Height-width ratio

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

figure('Position',[100,100, 900, 800]);

for t = 1:1050

    subplot(2, 2, 1);
    imagesc(reshape(data_normalized(:, t), nx, ny))
    colormap gray;
    title(t)
    axis off;

    subplot(2,2,2);
    imagesc(reshape(recon_og_clu{31}(:, t), nx, ny))
    colormap gray;
    axis off;

    subplot(2,2,3);
    imagesc(reshape(recon_og_clu{31}(:, t), nx, ny))
    colormap gray;
    axis off;

    subplot(2,2,4);
    imagesc(reshape(recon_og_clu{31}(:, t), nx, ny))
    colormap gray;
    axis off;

    pause(0.5);
end

%% Reconstruct Nidelva as a function of SF

num_timesteps = 1050;
num_repeats = 5;
ns = 20;
sensor_type = 'random';

% List of undersampling rates
us_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15];
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
save('results_Nidelva/Nidelva_20sensors_random_rerun.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length');

%% Reconstruct Nidelva as a function of SF

num_timesteps = 1050;
num_repeats = 1;
ns = 20;
sensor_type = 'qr';

% List of undersampling rates
us_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15];
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
save('results_Nidelva/Nidelva_20sensors_qr_rerun.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length');

%% DEVELOPMENT

%[U, S, V] = svd(data_normalized(:, :) - mean(data_normalized, 2), 'econ');

[U D V] = svd(data_normalized - mean(data_normalized, 2), 'econ'); 
y = diag(data_normalized - mean(data_normalized, 2)); 
%%
y( y < (optimal_SVHT_coef(1058/291600,0) * median(y)) ) = 0; 
Xhat = U * diag(y) * V';

%%

loglog(diag(D) / sum(diag(D)), 'ko');
hold on;
loglog(([1, 1000]), ([(optimal_SVHT_coef(1058/291600,0) * median(y)),(optimal_SVHT_coef(1058/291600,0) * median(y))]), 'b--')
%ylim([10e-10 1])

disp((optimal_SVHT_coef(1058/291600,0) * median(y)))
%% reconstruct original
data_og = U(:, 2:end) * D(2:end, 2:end) * V(:, 2:end)';

hfig = figure;
t = tiledlayout(1,length(modes), padding='none', TileSpacing='none');
modes = [1, 2, 3, 4, 5];

for mode = 1:length(modes)
    nexttile;
    imagesc(reshape(U(:, modes(mode)), nx, ny));
    xlabel(modes(mode));
    set(gca, 'XTick', [], 'YTick', [])
    colormap gray;
end

% Set additional properties
fname = 'output_Nidelva/modes';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.3; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);

% Export the figure
%print(hfig, fname, '-dpdf');%, '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%

data_900 = U(:, 1:900) * D(1:900, 1:900) * V(:, 1:900)';
data_300 = U(:, 1:200) * D(1:200, 1:200) * V(:, 1:200)';

%%
t = 886;
nx=540;
ny=540;

hfig = figure;
tl = tiledlayout(1, 3, Padding="compact", TileSpacing="compact");

nexttile;
imagesc(reshape(data_normalized(:, t) - mean(data_normalized, 2), nx, ny));
colormap gray;
xlabel('Original');
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(data_900(:, t), nx, ny));
colormap gray;
xlabel('$r = 900$');
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(data_300(:, t), nx, ny));
colormap gray;
xlabel('$r = 300$');
set(gca, 'XTick', [], 'YTick', [])

% Set additional properties
fname = 'output_Nidelva/modes';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.3; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);

% Export the figure
print(hfig, fname, '-dpdf');%, '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');