%loading datasets
data_ran = load('results_Re100\Re100_10sensors_random.mat');
data_qr = load('results_Re100\Re100_10sensors_QR.mat');

%% Run this to get original flow field
run ("load_data.m");

addpath ../../utils/

data_U = UALL;
data_V = VALL;
data_VS = sqrt(data_U.^2 + data_V.^2);
%% Defining params

recon_ran = data_ran.recon;
recon_qr = data_qr.recon;

ssim_ran = data_ran.mean_ssim;
mse_ran = data_ran.mean_residuals;
train_len_ran = data_ran.training_set_length;

ssim_qr = data_qr.mean_ssim;
mse_qr = data_qr.mean_residuals;
train_len_qr = data_qr.training_set_length;

%% Display some snapshots of reconstructed flow

% Determine the color limits based on original data
clim = [min(data_VS(:))-0.05, max(data_VS(:))+0.05];

% Choose timesteps to visualize
t = 7;

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(3,1, 'TileSpacing', 'tight', 'Padding', 'tight'); % Adjust spacing

% Reshape the flow fields for plotting
x_true = reshape(data_VS(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon_ran = reshape(recon_ran(:, t, 14), nx, ny); % Adjust nx and ny accordingly
x_recon_qr = reshape(recon_qr(:, t, 14), nx, ny); % Adjust nx and ny accordingly


% Plot true flow field
nexttile
imagesc(x_true);
xlabel('Original');
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Plot reconstructed flow field random
nexttile
imagesc(x_recon_ran);
xlabel(['Reconstructed (random)']);
hold on;
%plot(sensor_ran(:, 1), sensor_ran(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
hold off;
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Plot reconstructed flow field QR
nexttile
imagesc(x_recon_qr);
hold on;
% Plot sensors, convert linear indices to (row, col) coordinates
%[y_coords, x_coords] = ind2sub([nx, ny], sensor_qr);
%plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);

xlabel(['Reconstructed (QR)']);
colormap(cmap_);
caxis(clim);
%colorbar;
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 1.2; % Height-width ratio

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

%% Plot MSE vs undersampling_rate

hfig = figure;

plot(data_ran.results_table.UndersamplingRate, mse_ran, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Random sensors');
hold on;
plot(data_ran.results_table.UndersamplingRate, mse_qr, '+-', 'LineWidth', 1.5, 'DisplayName', 'QR sensors');
ylim([0, 1])
xlim([2 20])
ylabel('MSE')
xlabel('Undersampling frequency')
legend('Location','northeast');

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
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

%% Plot SSIM vs undersampling_rate

hfig = figure;

plot(data_ran.results_table.UndersamplingRate, ssim_ran, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Random sensors');
hold on;
plot(data_ran.results_table.UndersamplingRate, ssim_qr, '+-', 'LineWidth', 1.5, 'DisplayName', 'QR sensors');
ylim([0, 1])
xlim([2 20])
ylabel('SSIM')
xlabel('Undersampling frequency')
legend('Location','southeast');

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
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

%% Plot MSE vs training set length

hfig = figure;

semilogx(data_ran.results_table.TrainingSetLength, mse_ran, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Random sensors');
hold on;
semilogx(data_ran.results_table.TrainingSetLength, mse_qr, '+-', 'LineWidth', 1.5, 'DisplayName', 'QR sensors');
ylim([0, 1])
%xlim([5 20])
ylabel('SSIM')
xlabel('Training set length')
legend('Location','southeast');

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
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

semilogx(data_ran.results_table.TrainingSetLength, ssim_ran, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Random sensors');
hold on;
semilogx(data_ran.results_table.TrainingSetLength, ssim_qr, '+-', 'LineWidth', 1.5, 'DisplayName', 'QR sensors');
ylim([0, 1])
%xlim([5 20])
ylabel('SSIM')
xlabel('Training set length')
legend('Location','southeast');

% Set additional properties
fname = 'output/tester';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',21); % Adjust font size
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

%% Define the number of colors in the colormap
n_color = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n_color)', linspace(0, 1, n_color)', ones(n_color, 1); ...
        ones(n_color, 1), linspace(1, 0, n_color)', linspace(1, 0, n_color)'];