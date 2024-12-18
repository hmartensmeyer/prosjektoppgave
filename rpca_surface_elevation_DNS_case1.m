%Importing data

%data = load('../../data/eta.mat');
run '..'/Reconstruction/Surface_DNS/load_data.m

%% Normalization

% Min-Max Normalization per column
min_vals = min(data_reshaped, [], 1);
max_vals = max(data_reshaped, [], 1);
epsilon = 1e-8; % To avoid division by zero
data_reshaped = (data_reshaped - min_vals) ./ (max_vals - min_vals + epsilon);

%%

X = data_reshaped(:, 1:1050);
X_noiseless = data_reshaped(:,1:1050);

%% add salt & pepper
eta = 0.1; % to add noise change eta to fraction of occluded points. eta = 0.2 for 20%
if eta ~=0
    rep = std(X(:))*10;
    x = rand(size(X(1:end,:)));
    b = sort(x(:));
    thresh = b(floor(.5*eta*numel(b)));
    
    X(x<thresh) = rep;
    
    x = rand(size(X(1:end,:)));
    b = sort(x(:));
    thresh = b(floor(.5*eta*numel(b)));
    
    X(x<thresh) = -rep;
end

%%
addpath '../../utils/'
[L, S] = ALM_RPCA(X, 1, 1e-5, 1000);

%% Manual approach for tuning lambda parameter

[L_1p5, S_1p5] = ALM_RPCA(X, 1.5, 1e-5, 1000);
disp('----HERE----')
[L_2, S_2] = ALM_RPCA(X, 2, 1e-5, 1000);
disp('----HERE----')
[L_2p5, S_2p5] = ALM_RPCA(X, 2.5, 1e-5, 1000);
disp('----HERE----')
[L_3, S_3] = ALM_RPCA(X, 3, 1e-5, 1000);
disp('----HERE----')

%% Save all, X, L and S matrices to a .mat file
save('results_DNS/RPCA_results_1050_norm.mat', 'X_noiseless', 'X', 'L', 'S', 'L_1p5', 'S_1p5', 'L_2', 'S_2', 'L_2p5', 'S_2p5', 'L_3', 'S_3');

disp('All L and S matrices have been saved to RPCA_results.mat');

%% Loading saved .mat file

data_DNS = load('results_DNS\RPCA_results_1050_norm.mat');

%% defining properties

DNS_X_orig = data_DNS.X_noiseless;
DNS_L = data_DNS.L;
DNS_L_1p5 = data_DNS.L_1p5;
DNS_L_2 = data_DNS.L_2;
DNS_L_2p5 = data_DNS.L_2p5;
DNS_L_3 = data_DNS.L_3;
DNS_S = data_DNS.S;
DNS_X = data_DNS.X;
DNS_S_3 = data_DNS.S_3;


%% Colormap

% Define the number of colors in the colormap
n = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n)', linspace(0, 1, n)', ones(n, 1); ...
        ones(n, 1), linspace(1, 0, n)', linspace(1, 0, n)'];

%% Plot comparative figure
nx = 256;
ny = 256;
% Determine the color limits based on X_noiseless
clim = [min(X_noiseless(:)-0.1), max(X_noiseless(:)+0.1)];
%clim = [-1, 1];

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact'); % Adjust spacing

% Plot each component in a separate tile with consistent color limits
nexttile
imagesc(reshape(X_noiseless(:,55),nx,ny));
colormap(cmap_);
caxis(clim); % Apply color limits
%colorbar;
xlabel('Original')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(X(:,55),nx,ny)), colormap(cmap_)
caxis(clim); % Apply color limits
%colorbar;
xlabel('$\mathbf{X}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(DNS_L_3(:,55),nx,ny)), colormap(cmap_)
caxis(clim); % Apply color limits
xlabel('$\mathbf{L}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(DNS_S_3(:,55),nx,ny)), colormap(cmap_)
caxis([-1,1]); % Apply color limits
%colorbar;
xlabel('$\mathbf{S}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output_DNS/RPCA_DNS_SP10_10pc_lambda3';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.8; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',18); % Adjust font size
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

% %%
% 
% % Assuming X, L, and S are your data arrays with dimensions (num_points, num_timesteps)
% num_timesteps = 30; % Number of time steps
% grid_size = [256, 256]; % Dimensions of the grid
% 
% % Create a figure
% figure;
% 
% for t = 1:num_timesteps
%     % Reshape and plot X
%     subplot(2, 2, 1);
%     imagesc(reshape(X(:, t), grid_size)); 
%     colormap winter;
%     colorbar;
%     title('Original Image, X');
% 
%     % Reshape and plot L
%     subplot(2, 2, 3);
%     imagesc(reshape(L(:, t), grid_size)); 
%     colormap winter;
%     colorbar;
%     title('Low-Rank Approximation, L');
% 
%     % Reshape and plot S
%     subplot(2, 2, 4);
%     imagesc(reshape(S(:, t), grid_size)); 
%     colormap winter;
%     colorbar;
%     title('Sparse Component, S');
% 
%     % Hide the fourth subplot
%     subplot(2, 2, 2);
%     axis off;
% 
%     % Pause to create animation effect
%     pause(0.02); % Adjust the pause duration to control animation speed
% end
% 
% %%
% 
% % Assume X, L, S are of dimensions (Nx*Ny, Nt)
% % where Nx = 223, Ny = 196, and Nt is the number of time steps
% 
% % Parameters
% Nt = size(X, 2);
% fs = 15;  % Sampling frequency (adjust if necessary)
% N = 2^nextpow2(Nt);  % Ensure N is a power of 2 for FFT
% 
% % Reshape data for use in estpsd function
% %X_reshaped = reshape(X, Nx*Ny, Nt)';
% %L_reshaped = reshape(L, Nx*Ny, Nt)';
% %S_reshaped = reshape(S, Nx*Ny, Nt)';
% 
% % Calculate PSD for X, L, and S using the estpsd function
% [Gdat_noiseless, f_noiseless, e_r_noiseless] = estpsd(X_noiseless, N, fs);
% [Gdat_X, f_X, e_r_X] = estpsd(X, N, fs);
% [Gdat_L, f_L, e_r_L] = estpsd(L, N, fs);
% [Gdat_S, f_S, e_r_S] = estpsd(S, N, fs);
% 
% % Average PSD across all spatial points (channels)
% PSD_noiseless = mean(Gdat_noiseless, 2);
% PSD_X = mean(Gdat_X, 2);
% PSD_L = mean(Gdat_L, 2);
% PSD_S = mean(Gdat_S, 2);
% 
% % Plot the PSDs with publication-quality settings
% figure('Units', 'inches', 'Position', [0, 0, 6, 4]);
% 
% 
% % Plot each dataset with distinct line styles and markers
% loglog(f_noiseless, PSD_noiseless, 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Original noiseless data');
% hold on;
% loglog(f_X, PSD_X, 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'X (With noise)');
% loglog(f_L, PSD_L, 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'L (Low-Rank)');
% loglog(f_S, PSD_S, 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'S (Sparse)');
% ylim([1e-5, 1]);
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
% % Save as a PDF for publication
% print(fullfile(output_dir, 'PSD_surface_elev_SP_noise'), '-dpdf', '-r300');
% % Alternatively, save as EPS if preferred
% % print(fullfile(output_dir, 'psd_plot'), '-depsc2', '-r300');
% 
% 
% %%
% 
% % Parameters
% Nt = size(X, 2);
% fs = 1/0.02;          % Sampling frequency (Hz)
% N = 2^nextpow2(Nt / 2);  % Adjust segment length for desired smoothing
% 
% % Parameters for PSD estimation
% window_type = 'hann';  % Window function: 'hann', 'hamming', 'blackman', or 'rect'
% overlap = 0.5;         % 50% overlap
% 
% % Calculate PSD for X, L, and S using the modified estpsd function
% [Gdat_noiseless, f_noiseless, e_r_noiseless] = estpsd_smooth(X_noiseless, N, fs, window_type, overlap);
% [Gdat_X, f_X, e_r_X] = estpsd_smooth(X, N, fs, window_type, overlap);
% [Gdat_L, f_L, e_r_L] = estpsd_smooth(L, N, fs, window_type, overlap);
% [Gdat_S, f_S, e_r_S] = estpsd_smooth(S, N, fs, window_type, overlap);
% 
% % Average PSD across all spatial points (channels)
% PSD_noiseless = mean(Gdat_noiseless, 2);
% PSD_X_avg = mean(Gdat_X, 2);
% PSD_L_avg = mean(Gdat_L, 2);
% PSD_S_avg = mean(Gdat_S, 2);
% 
% % Plot the PSDs with publication-quality settings
% figure('Units', 'inches', 'Position', [0, 0, 6, 4]);
% 
% % Plot each dataset with distinct line styles and markers
% loglog(f_noiseless, PSD_noiseless, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Original noiseless data');
% hold on;
% loglog(f_X, PSD_X_avg, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'X (With noise)');
% loglog(f_L, PSD_L_avg, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'L (Low-Rank)');
% loglog(f_S, PSD_S_avg, 'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'S (Sparse)');
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
% % Save the figure (same as before)
% % [Include your saving code here]

% %% PSD estimation
% 
% % Parameters
% data_orig = X_noiseless;
% data_noise = X;
% data_L = L;
% data_S = S;
% fs = 30;        % Sampling frequency in Hz
% 
% % Define Welch's method parameters
% window_length = 512;            % Window length
% window = hann(window_length);    % Hanning window
% noverlap = 256;                  % 50% overlap
% nfft = 1024;                      % Number of FFT points
% 
% % Get the dimensions of the data
% [n, m] = size(data_orig);
% 
% % Initialize a variable to accumulate PSDs
% PSD_orig = zeros(nfft/2 + 1, 1);
% PSD_X = zeros(nfft/2 + 1, 1);
% PSD_L = zeros(nfft/2 + 1, 1);
% PSD_S = zeros(nfft/2 + 1, 1);
% 
% % Loop through all spatial points
% for i = 1:n
%     % Extract the velocity time series at (i, j)
%     original = squeeze(data_orig(i, :));
%     noise = squeeze(data_noise(i, :));
%     L_ = squeeze(data_L(i, :));
%     S_ = squeeze(data_S(i, :));
%     disp(i);
%     % Compute PSD using Welch's method
%     [P_orig, f_orig] = pwelch(original, window, noverlap, nfft, fs);
%     [P_X, f_X] = pwelch(noise, window, noverlap, nfft, fs);
%     [P_L, f_L] = pwelch(L_, window, noverlap, nfft, fs);
%     [P_S, f_S] = pwelch(S_, window, noverlap, nfft, fs);
% 
%     % Accumulate the PSD
%     PSD_orig = PSD_orig + P_orig;
%     PSD_X = PSD_X + P_X;
%     PSD_L = PSD_L + P_L;
%     PSD_S = PSD_S + P_S;
% end
% 
% % Calculate the average PSD
% 
% num_points = nx * ny;
% PSD_orig_ = PSD_orig / num_points;
% PSD_X_ = PSD_X / num_points;
% PSD_L_ = PSD_L / num_points;
% PSD_S_ = PSD_S / num_points;


%%

addpath '..'/utils/

%%

[PSD_orig, f_orig] = PSD_calc(DNS_X_orig, 1);
[PSD_L, f_L] = PSD_calc(DNS_L , 1);
[PSD_L_1p5, f_L_1p5] = PSD_calc(DNS_L_1p5, 1);
[PSD_L_2, f_L_2] = PSD_calc(DNS_L_2, 1);
[PSD_L_2p5, f_L_2p5] = PSD_calc(DNS_L_2p5, 1);
[PSD_L_3, f_L_3] = PSD_calc(DNS_L_3, 1);

[PSD_X, f_X] = PSD_calc(DNS_X, 1);
[PSD_S, f_S] = PSD_calc(DNS_S, 1);

%% PSD orig, X, L, S

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'k:', 'LineWidth', 1.5, 'Displayname', 'Original flow field');
hold on;
loglog(f_X, PSD_X, 'LineWidth', 1.5, 'Displayname', 'X');
loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L');
loglog(f_S, PSD_S, 'LineWidth', 1.5, 'Displayname', 'S');
%hold on;
%loglog(f_orig, f_orig.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (arbitrary unit)');
ylabel('Power Spectral Density');
xlim([0 1])
set(gca, 'XTickLabel', [], 'YTickLabel', []) %
grid on

fname = 'output_DNS/PSD_RPCA_DNS_SP10_10pc_lambda1';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.55; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20) % adjust fontsize to your document
legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector','-fillpage')
%print(hfig,fname,'-dpng','-vector')

%% PLOT DIFFERENT LAMBDA VALUES

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1$');
hold on;
loglog(f_L_1p5, PSD_L_1p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1.5$');
loglog(f_L_2, PSD_L_2, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2$');
loglog(f_L_2p5, PSD_L_2p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2.5$');
loglog(f_L_3, PSD_L_3, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 3$');
loglog(f_orig, PSD_orig, 'k:', 'LineWidth', 1.5, 'Displayname', 'Original flow field', 'MarkerSize', 5);
%loglog(f_orig, f_orig.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (arbitrary unit)');
ylabel('Power Spectral Density');
grid on;
%ylim([1e-4 1e5])
set(gca, 'XTickLabel', [], 'YTickLabel', []) %
grid on

fname = 'output_DNS/PSD_DNS_lambda_tuning';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.55; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
legend('Location', 'southwest', 'FontSize', 21, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');
%print(hfig,fname,'-dpng','-vector')

%% Plot and save error

addpath '../utils/'

[res, ssim] = error_calc(DNS_X_orig, DNS_L, 0, false);

disp(res);
disp(ssim);

%%

fprintf('res: %.10f\n', res);
fprintf('ssim: %.10f\n', ssim);


%%

disp(rank(L_2))

%%

% Calculate PSNR
[psnr_value, ~] = psnr(DNS_X_orig, DNS_L_3);

% Display PSNR value
fprintf('The PSNR value is: %.2f dB\n', psnr_value);