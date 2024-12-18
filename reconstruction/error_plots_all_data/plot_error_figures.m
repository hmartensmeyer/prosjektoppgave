%loading datasets
data_ran_DNS = load('../Surface_DNS/results_DNS\DNS_20sensors_random_full_rerun.mat');
data_qr_DNS = load('../Surface_DNS/results_DNS\DNS_20sensors_QR_full_rerun.mat');
%%
data_ran_Re7000 = load('../Cylinder_Re7000/results_Re7000\Re7000_20sensors_random_full.mat');
data_qr_Re7000 = load('../Cylinder_Re7000/results_Re7000\Re7000_20sensors_QR_full.mat');
%%
data_ran_Re100 = load('../Cylinder_Re100/results_Re100\Re100_10sensors_random.mat');
data_qr_Re100 = load('../Cylinder_Re100/results_Re100\Re100_10sensors_QR.mat');
%%
dara_ran_Nidelva = load('../Nidelva/results_Nidelva/Nidelva_20sensors_random_rerun.mat');
dara_qr_Nidelva = load('../Nidelva/results_Nidelva/Nidelva_20sensors_qr_rerun.mat');

%% Define the number of colors in the colormap
n_color = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n_color)', linspace(0, 1, n_color)', ones(n_color, 1); ...
        ones(n_color, 1), linspace(1, 0, n_color)', linspace(1, 0, n_color)'];
% %% Run this to get original flow field
% run ("load_data.m");
% 
% addpath ../../utils/

%% Defining params

% DNS with Random and QR Sensor Placement
recon_ran_DNS = data_ran_DNS.recon;
ssim_ran_DNS = data_ran_DNS.mean_ssim;
mse_ran_DNS = data_ran_DNS.mean_residuals;
train_len_ran_DNS = data_ran_DNS.training_set_length;

recon_qr_DNS = data_qr_DNS.recon;
ssim_qr_DNS = data_qr_DNS.mean_ssim;
mse_qr_DNS = data_qr_DNS.mean_residuals;
train_len_qr_DNS = data_qr_DNS.training_set_length;

% Re = 7000 with Random and QR Sensor Placement
recon_ran_Re7000 = data_ran_Re7000.recon;
ssim_ran_Re7000 = data_ran_Re7000.mean_ssim;
mse_ran_Re7000 = data_ran_Re7000.mean_residuals;
train_len_ran_Re7000 = data_ran_Re7000.training_set_length;

recon_qr_Re7000 = data_qr_Re7000.recon;
ssim_qr_Re7000 = data_qr_Re7000.mean_ssim;
mse_qr_Re7000 = data_qr_Re7000.mean_residuals;
train_len_qr_Re7000 = data_qr_Re7000.training_set_length;

% Re = 100 with Random and QR Sensor Placement
recon_ran_Re100 = data_ran_Re100.recon;
ssim_ran_Re100 = data_ran_Re100.mean_ssim;
mse_ran_Re100 = data_ran_Re100.mean_residuals;
train_len_ran_Re100 = data_ran_Re100.training_set_length;

recon_qr_Re100 = data_qr_Re100.recon;
ssim_qr_Re100 = data_qr_Re100.mean_ssim;
mse_qr_Re100 = data_qr_Re100.mean_residuals;
train_len_qr_Re100 = data_qr_Re100.training_set_length;

% Nidelva with Random and QR Sensor Placement
%recon_ran_Nidelva = data_ran_S.recon;
ssim_ran_Nidelva = dara_ran_Nidelva.mean_ssim;
mse_ran_Nidelva = dara_ran_Nidelva.mean_residuals;
train_len_ran_Nidelva = dara_ran_Nidelva.training_set_length;

%recon_qr_Nidelva = data_qr_Nidelva.recon;
ssim_qr_Nidelva = dara_qr_Nidelva.mean_ssim;
mse_qr_Nidelva = dara_qr_Nidelva.mean_residuals;
train_len_qr_Nidelva = dara_qr_Nidelva.training_set_length;


% %% Display some snapshots of reconstructed flow
% 
% % Determine the color limits based on original data
% clim = [min(data_normalized(:))-0.05, max(data_normalized(:))+0.05];
% 
% % Choose timesteps to visualize
% t = 55;
% 
% % Set up the figure and tiled layout
% hfig = figure;
% tiledlayout(1,3, 'TileSpacing', 'compact', 'Padding', 'tight'); % Adjust spacing
% 
% % Reshape the flow fields for plotting
% x_true = reshape(data_normalized(:, t), nx, ny);     % Adjust nx and ny accordingly
% x_recon_ran = reshape(recon_ran(:, t, 9), nx, ny); % Adjust nx and ny accordingly
% x_recon_qr = reshape(recon_qr(:, t, 9), nx, ny); % Adjust nx and ny accordingly
% 
% 
% % Plot true flow field
% nexttile(1)
% imagesc(x_true);
% xlabel('Original');
% colormap(cmap_);
% caxis(clim);
% %colorbar;
% set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks
% 
% % Plot reconstructed flow field random
% nexttile(2)
% imagesc(x_recon_ran);
% xlabel(['Random']);
% hold on;
% %plot(sensor_ran(:, 1), sensor_ran(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
% hold off;
% colormap(cmap_);
% caxis(clim);
% %colorbar;
% set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks
% 
% % Plot reconstructed flow field QR
% nexttile(3)
% imagesc(x_recon_qr);
% hold on;
% % Plot sensors, convert linear indices to (row, col) coordinates
% %[y_coords, x_coords] = ind2sub([nx, ny], sensor_qr);
% %plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
% 
% xlabel('QR');
% colormap(cmap_);
% caxis(clim);
% %colorbar;
% set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks
% 
% % Set additional properties
% fname = 'output_DNS/tester';
% picturewidth = 20; % Set figure width in centimeters
% hw_ratio = 0.35; % Height-width ratio
% 
% set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
% set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% 
% % Configure printing options
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
% box on;
% 
% % Export the figure
% %print(hfig, fname, '-dpdf', '-vector', '-fillpage');
% %print(hfig, fname, '-dpng', '-vector');

%% Plot MSE vs undersampling_rate

hfig = figure;

% Plot MSE vs Undersampling Rate for DNS with Random and QR Sensors
h3 = plot(data_ran_DNS.results_table.UndersamplingRate(1:14), data_ran_DNS.mean_residuals(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (random)', 'MarkerSize',4);
hold on;
h4 = plot(data_qr_DNS.results_table.UndersamplingRate(1:14), data_qr_DNS.mean_residuals(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (QR)', 'MarkerSize',4);

% Plot MSE vs Undersampling Rate for Re7000 with Random and QR Sensors
h1 = plot(data_ran_Re7000.results_table.UndersamplingRate(1:14), data_ran_Re7000.mean_residuals(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (random)', 'MarkerSize',4);
h2 = plot(data_qr_Re7000.results_table.UndersamplingRate(1:14), data_qr_Re7000.mean_residuals(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (QR)', 'MarkerSize',4);

% Plot MSE vs Undersampling Rate for Re100 with Random and QR Sensors
h5=plot(data_ran_Re100.results_table.UndersamplingRate(1:14), data_ran_Re100.mean_residuals(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (random)', 'MarkerSize',4);
h6=plot(data_qr_Re100.results_table.UndersamplingRate(1:14), data_qr_Re100.mean_residuals(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (QR)', 'MarkerSize',4);

% Plot MSE vs Undersampling Rate for Re100 with Random and QR Sensors
h7=plot(dara_ran_Nidelva.results_table.UndersamplingRate, dara_ran_Nidelva.mean_residuals, 'o--', 'LineWidth', 1.5, 'DisplayName', 'Nidelva (random)', 'MarkerSize',4);
h8=plot(dara_qr_Nidelva.results_table.UndersamplingRate, dara_qr_Nidelva.mean_residuals, 's-', 'LineWidth', 1.5, 'DisplayName', 'Nidelva (QR)', 'MarkerSize',4);
%legend;

xlim([2, 15.5])
%ylim([0, 0.68])

xlabel('Subsampling interval, $\beta$');
ylabel('NRMSR')

% % Collect all plot handles
% plotHandles = [h1, h2, h3, h4, h5, h6];
% 
% % Now, specify the order you want in the legend
% % For example, if you want to group all Random Sensors first, then QR Sensors:
% legendOrder = [h1, h2, h3, h4, h5, h6];
% legendLabels = {'Cylinder, Re7000 (random)', 'Cylinder, Re7000 (QR)', 'Free-surface DNS (random)', ...
%                 'Free-surface DNS (QR)', 'Cylinder, Re7000 (QR)', 'Cylinder, Re100 (QR)'};

% Set x-ticks at intervals of 0.1
%yticks(0:0.05:0.35);
%xticks(2:3:16);

% Set additional properties
fname = 'output_errors/MSE_three_datasets_US_legendoutside_meansubNid_NRMSR';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.5; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','eastoutside', 'FontSize', 16);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
grid on;

% Export the figure
print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% Plot SSIM vs undersampling_rate

hfig = figure;

% Plot SSIM vs Undersampling Rate for DNS with Random and QR Sensors
h3 = plot(data_ran_DNS.results_table.UndersamplingRate(1:14), data_ran_DNS.mean_ssim(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (random)', 'MarkerSize',4);
hold on;
h4=plot(data_qr_DNS.results_table.UndersamplingRate(1:14), data_qr_DNS.mean_ssim(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (QR)', 'MarkerSize',4);

% Plot SSIM vs Undersampling Rate for Re7000 with Random and QR Sensors
h1=plot(data_ran_Re7000.results_table.UndersamplingRate(1:14), data_ran_Re7000.mean_ssim(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (random)', 'MarkerSize',4);
h2=plot(data_qr_Re7000.results_table.UndersamplingRate(1:14), data_qr_Re7000.mean_ssim(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (QR)', 'MarkerSize',4);

% Plot SSIM vs Undersampling Rate for Re100 with Random and QR Sensors
h5=plot(data_ran_Re100.results_table.UndersamplingRate(1:14), data_ran_Re100.mean_ssim(1:14), 'o--', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (random)', 'MarkerSize',4);
h6=plot(data_qr_Re100.results_table.UndersamplingRate(1:14), data_qr_Re100.mean_ssim(1:14), 's-', 'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (QR)', 'MarkerSize',4);

% Plot MSE vs Undersampling Rate for Re100 with Random and QR Sensors
h7=plot(dara_ran_Nidelva.results_table.UndersamplingRate, dara_ran_Nidelva.mean_ssim, 'o--', 'LineWidth', 1.5, 'DisplayName', 'Nidelva (random)', 'MarkerSize',4);
h8=plot(dara_qr_Nidelva.results_table.UndersamplingRate, dara_qr_Nidelva.mean_ssim, 's-', 'LineWidth', 1.5, 'DisplayName', 'Nidelva (QR)', 'MarkerSize',4);


%ylim([0, 1.05])
xlim([2 15.5])
ylabel('SSIM')
xlabel('Subsampling interval, $\beta$')
%legend('Location','southeast');
%xticks(2:3:16);

% % Collect all plot handles
% plotHandles = [h1, h2, h3, h4, h5, h6];
% 
% % Now, specify the order you want in the legend
% % For example, if you want to group all Random Sensors first, then QR Sensors:
% legendOrder = [h4, h3, h6, h5, h2, h1];
% legendLabels = {'Cylinder, Re7000 (random)', 'Cylinder, Re7000 (QR)', 'Free-surface DNS (random)', ...
%                 'Free-surface DNS (QR)', 'Cylinder, Re7000 (QR)', 'Cylinder, Re100 (QR)'};

% Set additional properties
fname = 'output_errors/SSIM_three_datasets_US_legendoutside_meansubNid';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','eastoutside', 'FontSize', 17);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
grid on;

% Export the figure
print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% Plot MSE vs training set length

hfig = figure;

% Plot MSE for Re7000 with Random and QR Sensors
h1 = semilogx(data_ran_Re7000.training_set_length, data_ran_Re7000.mean_residuals, 'o--', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (Random)', 'MarkerSize',4);
hold on;
h2 = semilogx(data_qr_Re7000.training_set_length, data_qr_Re7000.mean_residuals, 's-', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (QR)', 'MarkerSize',4);

% Plot MSE for DNS with Random and QR Sensors
h3 = semilogx(data_ran_DNS.training_set_length, data_ran_DNS.mean_residuals, 'd--', ...
    'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (Random)', 'MarkerSize',4);
h4 = semilogx(data_qr_DNS.training_set_length, data_qr_DNS.mean_residuals, 'p-', ...
    'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (QR)', 'MarkerSize',4);

% Plot MSE for Re100 with Random and QR Sensors
h5 = semilogx(data_ran_Re100.training_set_length, data_ran_Re100.mean_residuals, 'v--', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (Random)', 'MarkerSize',4);
h6 = semilogx(data_qr_Re100.training_set_length, data_qr_Re100.mean_residuals, '^-', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (QR)', 'MarkerSize',4);

% Set plot limits and labels
%ylim([0, 1]);
ylabel('MSE');
xlabel('Training Set Length');
% xlim([5 20]); % Uncomment if you want to set x-axis limits

% Customize legend
legendOrder = [h1, h2, h3, h4, h5, h6];
legendLabels = {
    'Cylinder, Re7000 (Random)', 'Cylinder, Re7000 (QR)', 
    'Free-surface DNS (Random)', 'Free-surface DNS (QR)', 
    'Cylinder, Re100 (Random)', 'Cylinder, Re100 (QR)'
};

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
legend(legendOrder, legendLabels, 'Location','southwest', 'FontSize', 17);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
grid on;

% Export the figure
%print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');


%% Plot SSIM vs training set length

hfig = figure;

% Plot MSE for Re7000 with Random and QR Sensors
h1 = semilogx(data_ran_Re7000.training_set_length, data_ran_Re7000.mean_ssim, 'o--', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (Random)', 'MarkerSize',4);
hold on;
h2 = semilogx(data_qr_Re7000.training_set_length, data_qr_Re7000.mean_ssim, 's-', ...
    'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re7000 (QR)', 'MarkerSize',4);

% Plot MSE for DNS with Random and QR Sensors
h3 = semilogx(data_ran_DNS.training_set_length, data_ran_DNS.mean_ssim, 'd--', ...
    'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (Random)', 'MarkerSize',4);
hold on;
h4 = semilogx(data_qr_DNS.training_set_length, data_qr_DNS.mean_ssim, 'p-', ...
    'LineWidth', 1.5, 'DisplayName', 'Free-surface DNS (QR)', 'MarkerSize',4);

% Plot MSE for Re100 with Random and QR Sensors
h5 = semilogx(data_ran_Re100.training_set_length, data_ran_Re100.mean_ssim, 'v--', ...
   'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (Random)', 'MarkerSize',4);
h6 = semilogx(data_qr_Re100.training_set_length, data_qr_Re100.mean_ssim, '^-', ...
   'LineWidth', 1.5, 'DisplayName', 'Cylinder, Re100 (QR)', 'MarkerSize',4);

% Set plot limits and labels
%ylim([0, 1]);
ylabel('SSIM');
xlabel('Training Set Length');
% xlim([5 20]); % Uncomment if you want to set x-axis limits

% Customize legend
legendOrder = [h1, h2, h3, h4, h5, h6];
legendLabels = {
    'Cylinder, Re7000 (Random)', 'Cylinder, Re7000 (QR)', 
    'Free-surface DNS (Random)', 'Free-surface DNS (QR)', 
    'Cylinder, Re100 (Random)', 'Cylinder, Re100 (QR)'
};

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
%legend(legendOrder, legendLabels, 'Location','southwest', 'FontSize', 17);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
grid on;

% Export the figure
%print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');