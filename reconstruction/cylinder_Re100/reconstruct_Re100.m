run('load_data.m');

%%
num_timesteps = 20;
num_repeats = 1;
ns = 10;
sensor_type = 'random';
SF = 9;
rescale = true;

%%
addpath ../../utils/

data_U = UALL;
data_V = VALL;
data_VS = sqrt(data_U.^2 + data_V.^2);

[recon_ran, res_ran, ssim_ran, sensor_ran] = SPARSE_RECONSTRUCTION_meansub(data_VS, flow, num_timesteps, num_repeats, ns, sensor_type, SF, rescale, nx, ny);
[recon_qr, res_qr, ssim_qr, sensor_qr] = SPARSE_RECONSTRUCTION_meansub(data_VS, flow, num_timesteps, num_repeats, ns, 'qr', SF, rescale, nx, ny);


%% Define the number of colors in the colormap
n_color = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n_color)', linspace(0, 1, n_color)', ones(n_color, 1); ...
        ones(n_color, 1), linspace(1, 0, n_color)', linspace(1, 0, n_color)'];

%% Display some snapshots of reconstructed flow

% Determine the color limits based on original data
clim = [min(data_VS(:))-0.05, max(data_VS(:))+0.05];

% Choose timesteps to visualize
t = 14;

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'tight'); % Adjust spacing

% Reshape the flow fields for plotting
x_true = reshape(data_VS(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon_ran = reshape(recon_ran(:, t), nx, ny); % Adjust nx and ny accordingly
x_recon_qr = reshape(recon_qr(:, t), nx, ny); % Adjust nx and ny accordingly

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
plot(sensor_ran(:, 1), sensor_ran(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
hold off;
colormap(cmap_);
%caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Plot reconstructed flow field QR
nexttile(4)
imagesc(x_recon_qr);
hold on;
% Plot sensors, convert linear indices to (row, col) coordinates
[y_coords, x_coords] = ind2sub([nx, ny], sensor_qr);
plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
xlabel(['Reconstructed (QR)']);
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output/illustration_Re100_slim';
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
print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%
addpath ../../utils/

[res_ran, ssim_ran] = error_calc(x_true, x_recon_ran, 1, false);
[res_qr, ssim_qr] = error_calc(x_true, x_recon_qr, 1, false);

disp(res_ran)
disp(res_qr)

%%
num_timesteps = 151;
num_repeats = 1;
ns = 10;
sensor_type = 'qr';
SF = 10;
rescale = true;

% List of undersampling rates
us_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
residuals_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
ssim_ = zeros(length(us_values), num_timesteps); % To store mean residuals for each us
recon = zeros(n, num_timesteps, length(us_values)); % (n, m, us)
training_set_length = zeros(length(us_values), 1);

for i = 1:length(us_values)
    [Reconstructed_Flow, residuals, ssim] = SPARSE_RECONSTRUCTION_meansub(data_VS, flow, num_timesteps, num_repeats, ns, sensor_type, us_values(i), rescale, nx, ny);

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
save('results_Re100/Re100_10sensors_QR.mat', 'recon', 'residuals_', 'ssim_', 'mean_residuals', 'mean_ssim', 'results_table', 'training_set_length');

%%

plot(us_values, mean_ssim);

%%
plot(us_values, mean_residuals);
%ylim ([0, 1]);

%%
plot(training_set_length, mean_residuals);













% %% Plot snapshots for "dictionary" cartoon
% for t=1:7:57
%     disp(t);
% 
%     x = UALL(:, t);
%     plotCylinder(reshape(x, nx, ny), flow);
%     axis off
%     % if (SAVE_FIGS)
%     %     saveas(gcf, sprintf('figs/FIG1_atom%d.svg', t))
%     % end
% end
% 
% %% Plot example figure with point measurements
% % Measurement: random pixels
% ns = 10;  % Number of point measurements
% 
% % Restrict to cylinder wake: final 80% of width, middle 50% of height
% sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
% sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
% % Convert to measurement matrix
% C = spdiags(ones(n, 1), 0, n, n);
% C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix
% 
% % Example flow snapshot
% x = UALL(:, 23);  
% plotCylinder(reshape(x, nx, ny), flow); hold on
% scatter(sensor_idx(:, 1), sensor_idx(:, 2), 50, 'k', 'filled'); axis off
% set(gcf,'Position',[100 100 600 260])
% % if (SAVE_FIGS)
% %     saveas(gcf, 'figs/FIG1_sensors.svg')
% % end
% 
% % Show training dictionary
% mTrain = 32;
% D = C*UALL(:, 1:5:mTrain);    % Measured library (includes mean for visualization)
% disp(size(D))
% figure()
% pcolor([D; D(end, :)]);
% colormap(flow.cmap); caxis(flow.clim);
% set(gcf,'Position',[1000 100 650 200]); axis off;
% % if (SAVE_FIGS)
% %     saveas(gcf, 'figs/FIG1_dict.svg')
% % end
% 
% 
% %% Test image examples
% Train = UALL(:, 1:5:151) - mean(UALL, 2);  % Some snapshots are skipped so aspect ratio of library is good
% rms_vort = flow.avg_energy;
% 
% x = UALL(:, 23);
% 
% D = C*Train;  % Measured library (now mean-subtracted)
% 
% disp(size(x))
% disp(size(D));
% 
% %%
% 
% % Add noise to the measurement
% eta = 0.65;
% noise = 0*eta*rms_vort*randn([n, 1]);
% y = C*(x);
% disp(size(y));
% 
% %%
% 
% % Compute sparse approximation to the entire flow field and calculate error
% %s = sp_approx(y, D, eta, flow);
% s = sp_approx(y, D, 1e-9); %for LASSO
% [x_hat, res] = reconstruct(x, Train, s, flow, true);
% disp(res)
% 
% %% Plot results
% 
% figure()
% plotCylinder(reshape(x_hat+flow.mean_flow, nx, ny), flow); axis off
% set(gcf,'Position',[100 100 600 260]);
% title('xhat')
% 
% % if (SAVE_FIGS)
% %     saveas(gcf, 'figs/FIG1_xhat.svg')
% % end


%% Try to reconstruct the full flow field

% Define the flows
mean_flow = mean(UALL, 2);
Train = UALL(:, 1:5:end) - mean_flow;

% % Testing random sampling of snapshots
% num_snapshots = size(UALL, 2);
% % Randomly select 50% of the columns
% random_indices = randperm(num_snapshots, round(0.8 * num_snapshots));
% % Sample the selected snapshots from the training set
% Train = UALL(:, random_indices); % Randomly sampled 50% of snapshots

% Measurement: random pixels
ns = 10; % Number of point measurements
% Restrict to cylinder wake: final 80% of width, middle 50% of height
sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
% Convert to measurement matrix
C = spdiags(ones(n, 1), 0, n, n);
C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

% %Try to make sensing from QR-sensors
% % Compute the SVD of the snapshot matrix for QR-based measurement selection
% [U, ~, ~] = svd(Train, 'econ');
% k = 20;  % Number of leading modes to consider
% U_k = U(:, 1:k);
% 
% % Perform QR decomposition with column pivoting on U_k'
% [~, ~, E] = qr(U_k', 'vector');
% E = E';  % Transpose to make it a column vector
% 
% % QR-factorization measurements
% % Sensor indices selected via QR-factorization
% linear_idx_qr = E(1:ns);
% 
% % Construct the measurement matrix C_qr
% C = sparse(1:ns, linear_idx_qr, ones(ns, 1), ns, n);

% Initialize matrix to store reconstructed flow fields
Reconstructed_Flows = zeros(n, 60);
residuals = zeros(60, 1);
num_repeats = 1;  % Number of times to repeat reconstruction per timestep

for t = 1:60
    % True flow field at timestep t
    x = UALL(:, t) - mean_flow;

    % Initialize variables to accumulate results over repeats
    x_hat_sum = zeros(n, 1);  % Sum of reconstructed flow fields
    res_sum = 0;              % Sum of residuals
    
    for repeat = 1:num_repeats
        % Sensor measurements
        y = C * x;
        
        % Measured dictionary
        D = C * Train;
        
        s = sp_approx(y, D, 0, flow); %To use original script
        %s = sp_approx(y, D, 1e-2); %for LASSO

        disp(['s: ', num2str(size(s))]);
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
    disp(['Timestep ', num2str(t), ': Average residual over ', num2str(num_repeats), ' repeats = _______________________________________________', num2str(res_avg)]);
end

%%

% % Save the residuals array to a .mat file in the output folder
% save(fullfile('re7000-cylinder/output', 're_100_res_dt5_sens5.mat'), 'residuals');

%% Display errors of reconstructed flow

plot(residuals);
disp(mean(residuals));

%%

disp(norm(x-x_hat))
disp(norm(x+mean_flow))

%%

for t = 1:size(Reconstructed_Flows, 2)
    subplot(1,2,1);
    imagesc(reshape(UALL(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Original - Timestep ', num2str(t)]);
    
    subplot(1,2,2);
    imagesc(reshape(Reconstructed_Flows(:, t), nx, ny)); 
    colorbar; 
    %caxis([flow_min, flow_max]);
    title(['Reconstructed - Timestep ', num2str(t)]);
    colormap jet;
    drawnow; pause(0.1);
end


%% Display some snapshots of reconstructed flow

% Choose timesteps to visualize
timesteps_to_plot = [10, 50, 100, 150];

for i = 1:length(timesteps_to_plot)
    t = timesteps_to_plot(i);
    
    % Reshape the flow fields for plotting
    x_true = reshape(UALL(:, t), nx, ny);     % Adjust nx and ny accordingly
    x_recon = reshape(Reconstructed_Flows(:, t), nx, ny); % Adjust nx and ny accordingly
    
    % Plot true flow field
    figure;
    subplot(1,2,1);
    imagesc(x_true);
    title(['True Flow Field at Timestep ', num2str(t)]);
    colorbar;
    
    % Plot reconstructed flow field
    subplot(1,2,2);
    imagesc(x_recon);
    title(['Reconstructed Flow Field at Timestep ', num2str(t)]);
    colorbar;
end

%%

% List of undersampling rates
us_values = [2, 5, 10, 20, 30];
mean_residuals = zeros(length(us_values), 1); % To store mean residuals for each us

% Number of point measurements
ns = 10;

% Loop over each undersampling rate
for idx = 1:length(us_values)
    us = us_values(idx);
    
    % -----------------------------
    % Data Preparation
    % -----------------------------
    
    % Define the mean flow and training data
    mean_flow = mean(UALL, 2);
    Train = UALL(:, 1:us:end) - mean_flow; % us is the undersampling parameter
    
    % -----------------------------
    % Sensor Selection (QR-based or random)
    % -----------------------------
    
    % %Restrict to cylinder wake: final 80% of width, middle 50% of height
    % sensor_idx = [randperm(round(0.8*ny), ns)' randperm(round(0.5*nx), ns)']; % Choose sensors on restricted area
    % sensor_idx = [round(0.2*ny)+sensor_idx(:, 1) round(0.25*nx)+sensor_idx(:, 2)];  % Translate to wake
    % %Convert to measurement matrix
    % C = spdiags(ones(n, 1), 0, n, n);
    % C = C(sensor_idx(:, 2) + (nx-1)*sensor_idx(:, 1), :);  % Sparse measurement matrix

    % Compute the SVD of the snapshot matrix
    [U, ~, ~] = svd(Train, 'econ');
    U_k = U(:, :); % Use all available modes

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
        x = UALL(:, t) - mean_flow;
        
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
            %s = sp_approx(y, D, 1e-2); %for LASSO

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
        disp(['us = ', num2str(us), ', Timestep ', num2str(t), ': Residual___________________________________________________________________________ = ', num2str(res_avg)]);
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

%train_length = round([151, 151/2, 151/5, 151/10, 151/20, 151/30]);

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
save('re7000-cylinder/output/Re100_US_QR.mat', 'us_values', 'mean_residuals');
