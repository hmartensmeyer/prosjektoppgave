%loading datasets
data_ran = load('results_Re7000\Re7000_20sensors_random_full.mat');
data_qr = load('results_Re7000\Re7000_20sensors_QR_full.mat');

%% Define the number of colors in the colormap
n_color = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n_color)', linspace(0, 1, n_color)', ones(n_color, 1); ...
        ones(n_color, 1), linspace(1, 0, n_color)', linspace(1, 0, n_color)'];
%% Run this to get original flow field
run ("load_data.m");

addpath ../../utils/

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
clim = [min(velocity_magnitude_field_reshaped(:))-0.05, max(velocity_magnitude_field_reshaped(:))+0.05];

% Choose timesteps to visualize
t = 273;

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2,2, 'TileSpacing', 'tight', 'Padding', 'tight'); % Adjust spacing

% Reshape the flow fields for plotting
x_true = reshape(velocity_magnitude_field_reshaped(:, t), nx, ny);     % Adjust nx and ny accordingly
x_recon_ran = reshape(recon_ran(:, t, 9), nx, ny); % Adjust nx and ny accordingly
x_recon_qr = reshape(recon_qr(:, t, 9), nx, ny); % Adjust nx and ny accordingly


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
%plot(sensor_ran(:, 1), sensor_ran(:, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
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
%[y_coords, x_coords] = ind2sub([nx, ny], sensor_qr);
%plot(x_coords, y_coords, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'LineWidth', 1.5);

xlabel('Reconstructed (QR)');
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
ylabel('MSE')
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

%% Try to calculate PSD for original flow field vs random vs QR?

addpath '..'/utils/

[PSD_orig, f_orig] = PSD_calc(velocity_magnitude_field_reshaped, 15);
[PSD_ran, f_ran] = PSD_calc(recon_ran(:,:,9) , 15); %PSD of the random sensor case where US = 10
[PSD_qr, f_qr] = PSD_calc(recon_qr(:,:,9), 15); %PSD of the QR sensor case where US = 10

%% PLOT X_OG, random and QR

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'LineWidth', 1.5, 'Displayname', 'Original flow field');
hold on;
loglog(f_ran, PSD_ran, 'LineWidth', 1.5, 'Displayname', 'Random');
loglog(f_qr, PSD_qr, 'LineWidth', 1.5, 'Displayname', 'QR');
%loglog(f_orig, f_orig.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
grid on;
%ylim([1e-8 1])

fname = 'output_Re7000/PSD_reconstruction_US10_taller';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20) % adjust fontsize to your document
legend('Location', 'northeast', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

%% Try to calculate PSD for original flow field vs QR at different sampling rates?

[PSD_qr_2, f_qr_2] = PSD_calc(recon_qr(:,:,1), 15); %PSD of the QR sensor case where US = 2
[PSD_qr_5, f_qr_5] = PSD_calc(recon_qr(:,:,4), 15); %PSD of the QR sensor case where US = 5
[PSD_qr_10, f_qr_10] = PSD_calc(recon_qr(:,:,9), 15); %PSD of the QR sensor case where US = 10
[PSD_qr_15, f_qr_15] = PSD_calc(recon_qr(:,:,14), 15); %PSD of the QR sensor case where US = 15
[PSD_qr_20, f_qr_20] = PSD_calc(recon_qr(:,:,19), 15); %PSD of the QR sensor case where US = 20

%% PLOT X_OG vs QR at different sampling rates
% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'LineWidth', 1.5, 'Displayname', 'Original flow field');
hold on;
%loglog(f_qr_2, PSD_qr_2, 'LineWidth', 1.5, 'Displayname', 'Sampling rate = 2');
loglog(f_qr_5, PSD_qr_5, 'LineWidth', 1.5, 'Displayname', 'Sampling rate = 5');
loglog(f_qr_10, PSD_qr_10, 'LineWidth', 1.5, 'Displayname', 'Sampling rate = 10');
loglog(f_qr_15, PSD_qr_15, 'LineWidth', 1.5, 'Displayname', 'Sampling rate = 15');
loglog(f_qr_20, PSD_qr_20, 'LineWidth', 1.5, 'Displayname', 'Sampling rate = 20');
%hold on;
%loglog(f_orig, f_orig.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
grid on;
ylim([1e-6 1])

fname = 'output_Re7000/PSD_reconstruction_QR_UStuning_taller';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.65; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20) % adjust fontsize to your document
legend('Location', 'northeast', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')