% legacy

run("load_data.m");

%%
num_timesteps = 30;
num_repeats = 1;
ns = 30;
sensor_type = 'cluster';
SF = 10;

%%
% %% Testing the cluster sensor approach
% addpath ../../utils/
% 
% [recon_og, res_og, ssim_og, sensor_pos_og] = SPARSE_RECONSTRUCTION(data_normalized, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);
% 
% %[recon_og, res_og, ssim_og] = SPARSE_RECONSTRUCTION_meansub(data_normalized, flow, num_timesteps, num_repeats, ns, sensor_type, SF, 'false', nx, ny);
% % %%
% % [recon_rpca, res_rpca, ssim_rpca, sensor_pos_rpca] = SPARSE_RECONSTRUCTION(S, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);
% % %%
% % [recon_svd, res_svd, ssim_svd, sensor_pos_svd] = SPARSE_RECONSTRUCTION(data_reconstructed(:, 1:300), flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);
% 
% %% display 
% figure('Position', [100, 100, 1200, 400]); % Adjust the dimensions as needed
% 
% for t = 1:200
%     subplot(1,2,1);
%     imagesc(reshape(data_normalized(:, t), nx, ny));
%     colormap gray;
%     colorbar;
% 
%     subplot(1,2,2);
%     imagesc(reshape(recon_og(:, t), nx, ny));
%     hold on;
%     title(ssim_og(t))
%     % Plot sensors, convert linear indices to (row, col) coordinates
%     % [y_coords, x_coords] = ind2sub([nx, ny], sensor_pos_og);
%     % plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %QR
%     %plot(sensor_pos_og(:, 1), sensor_pos_og(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %RANDOM
%     colormap gray;
%     colorbar;
%     pause(0.02);
% end
% 
% %%
% 
% % disp(size(sensor_pos_og))
% % disp(sensor_pos_og)
% 
% disp(nnz(sensor_pos_og))
% 
% disp(20*20*30)
% 
% 
% %% Error measures
% 
% % PSNR calculation
% psnr_recon = psnr(recon_og, data_normalized(:, 1:50));
% 
% % Display results
% fprintf('Metrics for Original vs Reconstructed Flow Field:\n');
% fprintf('PSNR: %.2f dB\n', psnr_recon);
% 
% disp(mean(ssim_og))
% %disp(mean(ssim_rpca))
% %disp(mean(ssim_svd))
% 
% %%
% 
% plot(res_og)

%% SVD of data_reshaped and then throw away first mode

% Perform SVD on the reshaped data
[U, Sigma, V] = svd(data_normalized - mean(data_normalized, 2), 'econ');

%% Choose the number of modes to plot and reconstruct
num_modes = [1, 2, 3, 4, 5]; % Modes to visualize
hfig = figure;
% Create a tiled layout for better visualization
num_plots = length(num_modes); % Number of modes to plot
t = tiledlayout(1, num_plots, 'TileSpacing', 'compact', 'Padding', 'tight'); % One row
% Plot the selected modes
for k = 1:num_plots
nexttile; % Move to the next tile
mode = num_modes(k);
imagesc(reshape(U(:, mode), nx, ny)); % Reshape to spatial dimensions
colormap gray;
title(['Mode ', num2str(mode)]);
set(gca, 'XTick', [], 'YTick', [])
end
fname = 'output_DNS/PSD_DNS_lambda_tuning';
picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.3; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
legend('Location', 'southwest', 'FontSize', 21, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex')
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
%Saver
%print(hfig,fname,'-dpdf','-vector');
%print(hfig,fname,'-dpng','-vector')


%%

starter = 1;
ender = 809;

% Reconstruct the data using the first few modes
data_reconstructed = U(:, starter:ender) * Sigma(starter:ender, starter:ender) * V(:, starter:ender)';

% Verify reconstruction quality by plotting the first frame
figure;
imagesc(reshape(data_reconstructed(:, 100), nx, ny));
colormap gray;
colorbar;
title(['Reconstructed Frame 1 using ', num2str(ender-starter), ' modes']);

%%
% Extract singular values from the diagonal of Sigma
singular_values = diag(Sigma);

% Plot the singular values
figure;
loglog(singular_values / sum(singular_values), 'o-', 'LineWidth', 1.5);
xlabel('Index');
ylabel('Singular Value');
title('Singular Values of data\_reshaped');
grid on;

%% Normalization of SVD-recon

% Min-Max Normalization per column
min_vals = min(data_reconstructed, [], 1);
max_vals = max(data_reconstructed, [], 1);
epsilon = 1e-8; % To avoid division by zero
data_reconstructed = (data_reconstructed - min_vals) ./ (max_vals - min_vals + epsilon);

%% Verify reconstruction quality by plotting the first frame
figure;
imagesc(reshape(data_reconstructed(:, 100), nx, ny));
colormap gray;
colorbar;
title(['Reconstructed Frame 1 using ', num2str(ender-starter), ' modes']);

%% Testing to do RPCA before reconstruction

[L_1, S_1] = ALM_RPCA(data_normalized(:, 1:300), 1, 1e-5, 1000);
[L_04, S_04] = ALM_RPCA(data_normalized(:, 1:300), 0.4, 1e-5, 1000);

%% Normalization of S

% Min-Max Normalization per column
min_vals = min(S_1, [], 1);
max_vals = max(S_1, [], 1);
epsilon = 1e-8; % To avoid division by zero
S_1 = (S_1 - min_vals) ./ (max_vals - min_vals + epsilon);

% Min-Max Normalization per column
min_vals = min(S_04, [], 1);
max_vals = max(S_04, [], 1);
epsilon = 1e-8; % To avoid division by zero
S_04 = (S_04 - min_vals) ./ (max_vals - min_vals + epsilon);

%%
addpath ../../utils/

[recon_og, res_og, ssim_og, sensor_pos_og] = SPARSE_RECONSTRUCTION(data_normalized, flow, num_timesteps, num_repeats, ns, 'cluster', SF, nx, ny);

[recon_og_r, res_og_r, ssim_og_r] = SPARSE_RECONSTRUCTION(data_normalized, flow, num_timesteps, num_repeats, ns, 'random', SF, nx, ny);
%%
[recon_rpca_1, res_rpca_1, ssim_rpca_1, sensor_pos_rpca_1] = SPARSE_RECONSTRUCTION(S_1, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);

[recon_rpca_04, res_rpca_04, ssim_rpca_04, sensor_pos_rpca_04] = SPARSE_RECONSTRUCTION(S_04, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);
%%
[recon_svd, res_svd, ssim_svd, sensor_pos_svd] = SPARSE_RECONSTRUCTION(data_reconstructed, flow, num_timesteps, num_repeats, ns, 'cluster', SF, nx, ny);

[recon_svd_r, res_svd_r, ssim_svd_r, sensor_pos_svd_r] = SPARSE_RECONSTRUCTION(data_reconstructed, flow, num_timesteps, num_repeats, ns, 'random', SF, nx, ny);
%%
%[recon_rpca_02, res_rpca_02, ssim_rpca_02, sensor_pos_rpca_02] = SPARSE_RECONSTRUCTION(S_02, flow, num_timesteps, num_repeats, ns, sensor_type, SF, nx, ny);

%% display 
figure('Position', [100, 100, 1200, 400]); % Adjust the dimensions as needed

for t = 1:200
    subplot(1,2,1);
    imagesc(reshape(data_normalized(:, t), nx, ny));
    colormap gray;
    colorbar;
    
    subplot(1,2,2);
    imagesc(reshape(recon_og(:, t), nx, ny));
    hold on;
    title(ssim_svd(t))
    % Plot sensors, convert linear indices to (row, col) coordinates
    % [y_coords, x_coords] = ind2sub([nx, ny], sensor_pos_og);
    % plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %QR
    %plot(sensor_pos_og(:, 1), sensor_pos_og(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %RANDOM
    colormap gray;
    colorbar;
    clim([0, 1])
    pause(0.2);
end

%% Error measures

% % PSNR calculation
% psnr_recon = psnr(recon_og, data_normalized(:, 1:50));
% %psnr_recon_rpca = psnr(recon_rpca, S(:, 1:100));
% %psnr_recon_svd = psnr(recon_svd, data_reconstructed(:, 1:30));
% 
% % Display results
% fprintf('Metrics for Original vs Reconstructed Flow Field:\n');
% fprintf('PSNR: %.2f dB\n', psnr_recon);
% 
% %fprintf('\nMetrics for RPCA-Processed Flow Field:\n');
% %fprintf('PSNR: %.2f dB\n', psnr_recon_rpca);

disp(mean(ssim_og))
disp(mean(ssim_og_r))
%disp(mean(ssim_rpca_1))
%disp(mean(ssim_rpca_04))
disp(mean(ssim_svd))
disp(mean(ssim_svd_r))


%%

plot(ssim_og, 'DisplayName', 'orig')
hold on;
plot(ssim_og_r, 'DisplayName','orig small')
%plot(ssim_rpca_1, 'DisplayName', 'rpca')
%plot(ssim_rpca_04, 'Displayname', 'rpca 06')
%plot(ssim_svd, 'DisplayName', 'svd')
legend;

%% Display some snapshots of reconstructed flow

% Choose timesteps to visualize
t = 19;

% Reshape the flow fields for plotting
x_true = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon = reshape(recon_og(:, t), nx, ny); % Adjust nx and ny accordingly
x_true_r = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon_r = reshape(recon_og_r(:, t), nx, ny); % Adjust nx and ny accordingly
%x_true_rpca = reshape(S(:, t), nx, ny);
%x_recon_rpca = reshape(recon_rpca(:, t), nx, ny);
x_true_svd = reshape(data_reconstructed(:, t), nx, ny);
x_recon_svd = reshape(recon_svd(:, t), nx, ny);
x_true_svd_r = reshape(data_reconstructed(:, t), nx, ny);
x_recon_svd_r = reshape(recon_svd_r(:, t), nx, ny);

% Plot true flow field
hfig = figure;
tiledlayout(2,2, "TileSpacing","compact","Padding",'compact')

nexttile
imagesc(x_true);
title(['Original']);
colormap gray;
%clim([0, 0.7])
axis off;

% Plot reconstructed flow field
nexttile
imagesc(x_recon);
hold on;
title('Reconstructed (large sensors)');
% Plot sensors, convert linear indices to (row, col) coordinates
% [y_coords, x_coords] = ind2sub([nx, ny], sensor_pos);
% plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %QR
%plot(sensor_pos(:, 1), sensor_pos(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5); %RANDOM
colormap gray;
%clim([0, 0.7])
axis off;

%RPCA-applied
nexttile
imagesc(x_true_r);
hold on;
title('Original');
colormap gray;
axis off;
%clim([0, 0.7])

%Reconstructed after RPCA
nexttile
imagesc(x_recon_r);
title('Reconstructed (pixel sensors)')
colormap gray;
axis off;
%clim([0, 0.7])

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
t = 19;

% PSNR calculation
psnr_recon_cluster = psnr(recon_svd(:, t), data_reconstructed(:, t));
psnr_recon_random = psnr(recon_svd_r(:, t), data_reconstructed(:, t));

disp(psnr_recon_cluster)
disp(psnr_recon_random)

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

for i = 1:length(us_values)
    [Reconstructed_Flow, residuals, ssim] = SPARSE_RECONSTRUCTION(data_normalized, flow, num_timesteps, num_repeats, ns, sensor_type, us_values(i), nx, ny);

    recon(:, :, i) = Reconstructed_Flow; % Store (n, m) for each us
    residuals_(i, :) = residuals;
    ssim_(i, :) = ssim;
end


% Compute mean residuals and SSIM across timesteps for each undersampling rate
mean_residuals = mean(residuals_, 2); % Average over columns (timesteps)
mean_ssim = mean(ssim_, 2);           % Average over columns (timesteps)

% Create the results table
results_table = table(us_values', mean_residuals, mean_ssim, ...
    'VariableNames', {'UndersamplingRate', 'MeanResidual', 'MeanSSIM'});

% Display the table
disp(results_table);

























% 
% %%
% disp(size(Train))
% disp(size(Test))
% 
% % %% Plot snapshots for "dictionary" cartoon
% % mean_flow = mean(data_reshaped,2);
% % x_show = Test(:, 20);% + mean_flow;
% % x_show = reshape(x_show, [nx, ny]);
% % 
% % figure;
% % set(gcf,'Position',[100 100 600 400]);
% % imagesc(x_show);
% % colorbar;
% % colormap gray;
% % 
% % %% Plot example figure with point measurements
% % % Measurement: random pixels
% % ns = 20; % Number of point measurements
% % 
% % % Restrict to cylinder wake: final 80% of width, middle 50% of height
% % sensor_idx = [randperm(round(1*ny), ns)' randperm(round(1*nx), ns)']; % Choose sensors on restricted area
% % %sensor_idx = [round(1*ny)+sensor_idx(:, 1) round(1*nx)+sensor_idx(:, 2)];  % Translate to wake
% % 
% % % Convert to measurement matrix
% % C = spdiags(ones(n, 1), 0, n, n);
% % C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% % disp(['C: ', num2str(size(C))]);
% % 
% % % %Try to make sensing from QR-sensors
% % % % Compute the SVD of the snapshot matrix for QR-based measurement selection
% % % [U, ~, ~] = svd(data_reshaped(:, 1:10:end), 'econ');
% % % %k = 50;  % Number of leading modes to consider
% % % U_k = U(:, :);
% % % 
% % % % Perform QR decomposition with column pivoting on U_k'
% % % [~, ~, E] = qr(U_k', 'vector');
% % % E = E';  % Transpose to make it a column vector
% % % 
% % % % QR-factorization measurements
% % % % Sensor indices selected via QR-factorization
% % % linear_idx_qr = E(1:ns);
% % % 
% % % % Construct the measurement matrix C_qr
% % % C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
% % % 
% % % % Convert linear indices to (row, col) coordinates
% % % [y_coords, x_coords] = ind2sub([nx, ny], linear_idx_qr);
% % 
% % %%
% % figure;
% % set(gcf,'Position',[100 100 600 260]);
% % imagesc(x_show);
% % hold on
% % scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
% % %scatter(x_coords, y_coords, 50, 'black');
% % set(gcf,'Position',[100 100 600 400]);
% % colorbar;
% % colormap gray;
% % 
% % 
% % %%
% % %Train = data_reshaped(:, 1:5:end) - mean_flow;
% % 
% % x = Test(:, 20);% + mean_flow;
% % 
% % D = C*Train;
% % %D = C*flow_field;
% % 
% % y = C*x;
% % 
% % % Display the sizes of D and y
% % disp(['Size of D (should be [ns, m]): ', num2str(size(D))]);
% % disp(['Size of y (should be [ns, 1]): ', num2str(size(y))]);
% % 
% % % Compute sparse approximation to the entire flow field and calculate error
% % eta = 0.1;
% % s = sp_approx(y, D, eta, flow); %original script
% % %s = sp_approx(y, D, 1e-12);
% % disp(['s: ', num2str(size(s))]);
% % [x_hat, res] = reconstruct(x, Train, s, flow, true);
% % disp(res)
% % 
% % %% Plot results
% % 
% % figure();
% % imagesc(reshape(x_hat, nx, ny));
% % colorbar;
% % set(gcf,'Position',[100 100 600 400]);
% % title('xhat');
% % colormap gray;
% 
% %%
% 
% % disp(norm(x))
% % disp(norm(x_hat))
% % disp(norm(x + mean_flow))
% 
% %% Try to reconstruct the full flow field
% 
% % Define the flows
% mean_flow = mean(data_normalized, 2);
% %%
% Train = data_normalized(:, 1:10:end) - mean_flow; % ds is the undersample parameter
% 
% % % Testing random sampling of snapshots
% % num_snapshots = size(data_normalized, 2);
% % % Randomly select 50% of the columns
% % random_indices = randperm(num_snapshots, round(0.5 * num_snapshots));
% % % Sample the selected snapshots from the training set
% % Train = data_normalized(:, random_indices); % Randomly sampled 50% of snapshots
% 
% % Measurement: random pixels
% ns = 100; % Number of point measurements
% % % Restrict to cylinder wake: final 80% of width, middle 50% of height
% % sensor_idx = [randperm(round(ny), ns)' randperm(round(nx), ns)']; % Choose sensors on restricted area
% % %sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
% % % Convert to measurement matrix
% % C = spdiags(ones(n, 1), 0, n, n);
% % C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% 
% %Try to make sensing from QR-sensors
% % Compute the SVD of the snapshot matrix for QR-based measurement selection
% addpath '..\..\utils\'
% 
% %[U, ~, ~] = rsvd(Train, 100);
% %k = 20;  % Number of leading modes to consider
% %U_k = U(:, :);
% 
% % Perform QR decomposition with column pivoting on U_k'
% [~, ~, E] = qr(Train', 'vector');
% E = E';  % Transpose to make it a column vector
% 
% % QR-factorization measurements
% % Sensor indices selected via QR-factorization
% linear_idx_qr = E(1:ns);
% 
% % Construct the measurement matrix 
% C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
% 
% % Initialize matrix to store reconstructed flow fields
% Reconstructed_Flows = zeros(n, 50);
% residuals = zeros(50, 1);
% num_repeats = 1;  % Number of times to repeat reconstruction per timestep
% 
% for t = 1:50
%     % True flow field at timestep t
%     x = data_normalized(:, t) - mean_flow;
% 
%     % Initialize variables to accumulate results over repeats
%     x_hat_sum = zeros(n, 1);  % Sum of reconstructed flow fields
%     res_sum = 0;              % Sum of residuals
% 
%     for repeat = 1:num_repeats
%         % Sensor measurements
%         y = C * x;
%         %y = x;
% 
%         % Measured dictionary
%         D = C * Train;
%         %D = Train;
% 
%         eta=0;
%         s = sp_approx(y, D, eta, flow); %original script
%         %s = sp_approx(y, D, 1e-5);
%         %disp(['s: ', num2str(size(s))]);
%         [x_hat, res] = reconstruct(x, Train, s, flow, false);
% 
%         % Accumulate reconstructed flow fields and residuals
%         x_hat_sum = x_hat_sum + x_hat;
%         res_sum = res_sum + res;
%     end
% 
%     % Compute average reconstructed flow field and residual for timestep t
%     x_hat_avg = x_hat_sum / num_repeats;
%     res_avg = res_sum / num_repeats;
% 
%     % Add mean flow back to the reconstructed flow field
%     x_hat_avg_full = x_hat_avg + mean_flow;
% 
%     % Store the averaged reconstructed flow field and residual
%     Reconstructed_Flows(:, t) = x_hat_avg_full;
%     residuals(t) = res_avg;
% 
%     % Display progress
%     disp(['Timestep ', num2str(t), ': Average residual over ', num2str(num_repeats), ' repeats = -----------------------------------------------------------', num2str(res_avg)]);
% end
% 
% %% Display errors of reconstructed flow
% 
% plot(residuals);
% disp(mean(residuals));
% 
% %% Display some snapshots of reconstructed flow
% 
% % Choose timesteps to visualize
% timesteps_to_plot = [26,27,36,50];
% 
% for i = 1:length(timesteps_to_plot)
%     t = timesteps_to_plot(i);
% 
%     % Reshape the flow fields for plotting
%     x_true = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
%     x_recon = reshape(Reconstructed_Flows(:, t), nx, ny); % Adjust nx and ny accordingly
% 
%     % Plot true flow field
%     figure('Position', [100, 100, 1000, 300]);
%     subplot(1,2,1);
%     imagesc(x_true);
%     title(['True Flow Field at Timestep ', num2str(t)]);
%     colormap gray;
%     colorbar;
% 
%     % Plot reconstructed flow field
%     subplot(1,2,2);
%     imagesc(x_recon);
%     title(['Reconstructed Flow Field at Timestep ', num2str(t)]);
%     colormap gray;
%     colorbar;
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %%
% 
% % Convert linear indices to (row, col) coordinates
% [y_coords, x_coords] = ind2sub([nx, ny], linear_idx_qr);
% 
% figure;
% imagesc(reshape(data_normalized(:, 1), [nx, ny]));
% hold on;
% %scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
% scatter(x_coords, y_coords, 50, 'k', 'filled'); axis off
% 
% %%
% 
% for t = 1:size(Reconstructed_Flows, 2)
%     subplot(1,2,1);
%     imagesc(reshape(data_normalized(:, t), nx, ny));
%     colorbar;
%     %caxis([flow_min, flow_max]);
%     title(['Original - Timestep ', num2str(t)]);
%     %clim([0,1]);
% 
%     subplot(1,2,2);
%     imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
%     colorbar; 
%     %caxis([flow_min, flow_max]);
%     title(['Reconstructed - Timestep ', num2str(t)]);
%     %clim([0,1]);
%     colormap gray;
%     drawnow; pause(0.1);
% end
% 
% %% Estimate PSD
% 
% addpath '../../utils/'
% 
% x_orig = data_reshaped(:, :);
% x_rec = Reconstructed_Flows;
% 
% % Parameters
% Nt_orig = size(x_orig, 2);
% Nt_rec = size(x_rec, 2);
% fs = 15;  % Sampling frequency (adjust if necessary)
% N_orig = 2^nextpow2(Nt_orig);  % Ensure N is a power of 2 for FFT
% N_rec = 2^nextpow2(Nt_rec);
% 
% % Calculate PSD for X, L, and S using the estpsd function
% [Gdat_orig, f_orig, e_r_orig] = estpsd(x_orig, N_orig, fs);
% [Gdat_rec, f_rec, e_r_rec] = estpsd(x_rec, N_rec, fs);
% 
% % Average PSD across all spatial points (channels)
% PSD_orig = mean(Gdat_orig, 2);
% PSD_rec = mean(Gdat_rec, 2);
% 
% % Plot the PSDs with publication-quality settings
% figure('Units', 'inches', 'Position', [0, 0, 6, 4]);
% 
% % Plot each dataset with distinct line styles and markers
% loglog(f_orig, PSD_orig, 'b', 'LineWidth', 1.5, 'DisplayName', 'Original');
% hold on;
% loglog(f_rec, PSD_rec, 'k', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed');
% 
% % Customize labels and legend
% xlabel('Frequency (Hz)', 'FontSize', 14);
% ylabel('Power Spectral Density', 'FontSize', 14);
% legend('Location', 'best', 'FontSize', 12, 'Box', 'off');
% 
% % Customize grid and axes
% grid on;
% set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', ...
%          'XMinorTick', 'on', 'YMinorTick', 'on', 'TickLength', [0.02, 0.02]);
% 
% % Adjust axes limits if necessary
% axis tight;
% hold off;
% 
% % Save the figure in a high-resolution vector format
% % Define the output directory
% output_dir = 'output';
% if ~exist(output_dir, 'dir')
%     mkdir(output_dir);
% end
% 
% % Adjust the paper size to match the figure size
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperPosition', [0, 0, 6, 4]);
% set(gcf, 'PaperSize', [6, 4]);
% 
% %%
% 
% % List of undersampling rates
% us_values = [2, 5, 10, 20, 30];
% mean_residuals = zeros(length(us_values), 1); % To store mean residuals for each us
% 
% % Number of point measurements
% ns = 30;
% 
% % Loop over each undersampling rate
% for idx = 1:length(us_values)
%     us = us_values(idx);
% 
%     % -----------------------------
%     % Data Preparation
%     % -----------------------------
% 
%     % Define the mean flow and training data
%     mean_flow = (mean(data_normalized, 2));
%     Train = data_normalized(:, 1:us:end) - mean_flow; % us is the undersampling parameter
% 
%     % -----------------------------
%     % Sensor Selection (QR-based or random)
%     % -----------------------------
% 
%     % Restrict to cylinder wake: final 80% of width, middle 50% of height
%     % sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
%     % sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
%     % Convert to measurement matrix
%     % C = spdiags(ones(n, 1), 0, n, n);
%     % C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% 
%     % Compute the SVD of the snapshot matrix
%     addpath '..\..\utils\'
%     k=100;
%     [U, ~, ~] = rsvd(Train, k);
%     %U_k = U(:, :); % Use all available modes
% 
%     % Perform QR decomposition with column pivoting on U_k'
%     [~, ~, E] = qr(U', 'vector');
%     E = E'; % Transpose to make it a column vector
% 
%     % Sensor indices selected via QR-factorization
%     linear_idx_qr = E(1:ns);
% 
%     % Construct the measurement matrix C
%     C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);
% 
%     % -----------------------------
%     % Sparse Reconstruction
%     % -----------------------------
% 
%     % Initialize matrices to store reconstructed flows and residuals
%     Reconstructed_Flows = zeros(n, 60);
%     residuals = zeros(60, 1);
%     num_repeats = 1; % Number of repeats per timestep (can be adjusted)
% 
%     for t = 1:60
%         % True flow field at timestep t
%         x = data_normalized(:, t) - mean_flow;
% 
%         % Initialize variables to accumulate results over repeats
%         x_hat_sum = zeros(n, 1); % Sum of reconstructed flow fields
%         res_sum = 0;             % Sum of residuals
% 
%         for repeat = 1:num_repeats
%             % Sensor measurements
%             y = C * x;
% 
%             % Measured dictionary
%             D = C * Train;
% 
%             % Sparse approximation
%             eta = 0;
%             s = sp_approx(y, D, eta, flow); % Modify as needed based on your sp_approx function
%             %s = sp_approx(y, D, 1e-5); %for LASSO
% 
%             % Reconstruction
%             [x_hat, res] = reconstruct(x, Train, s, flow, true); % Modify as needed based on your reconstruct function
% 
%             % Accumulate results
%             x_hat_sum = x_hat_sum + x_hat;
%             res_sum = res_sum + res;
%         end
% 
%         % Compute average reconstructed flow field and residual
%         x_hat_avg = x_hat_sum / num_repeats;
%         res_avg = res_sum / num_repeats;
% 
%         % Add mean flow back to the reconstructed flow field
%         x_hat_avg_full = x_hat_avg + mean_flow;
% 
%         % Store the reconstructed flow and residual
%         Reconstructed_Flows(:, t) = x_hat_avg_full;
%         residuals(t) = res_avg;
% 
%         % Display progress
%         disp(['us = ', num2str(us), ', Timestep ', num2str(t), ': Residual---------------------------------------------------------------- = ', num2str(res_avg)]);
%     end
% 
%     % % Compute and store the mean residual for this undersampling rate
%     % valid_residuals = residuals(~isnan(residuals));
%     % if ~isempty(valid_residuals)
%     %     mean_residual = mean(valid_residuals);
%     % else
%     %     mean_residual = NaN;
%     % end
%     % mean_residuals(idx) = mean_residual;
%     % disp(['us = ', num2str(us), ', Mean residual over all timesteps: ', num2str(mean_residual)]);
%     mean_residuals(idx) = mean(residuals);
% end
% 
% % Save the array of mean residuals as a function of undersampling rates
% results_table = table(us_values', mean_residuals, 'VariableNames', {'UndersamplingRate', 'MeanResidual'});
% disp(results_table);
% 
% %%
% 
% train_length = round([200, 100, 200/5, 200/10, 10, 200/30]);
% 
% % Plot the mean residuals versus undersampling rates
% figure;
% plot(us_values, mean_residuals, '-o', 'LineWidth', 2);
% xlabel('Undersampling Rate (us)');
% ylabel('Mean Residual');
% grid on;
% 
% % figure;
% % plot(train_length, mean_residuals, '-o', 'LineWidth', 2);
% % xlabel('Training set length');
% % ylabel('Mean Residual');
% % grid on;
% 
% %%
% 
% % Save the undersampling rates and corresponding mean residuals to a MAT-file
% save('output/nidelva_US_QR.mat', 'us_values', 'mean_residuals');
% 
% 
% %%
% 
% disp(['flow.avg_energy: ', num2str(flow.avg_energy)]);
% disp(['eps: ', num2str(eps)]);
