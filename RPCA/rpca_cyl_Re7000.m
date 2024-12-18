% Load the MATLAB file for vortex shedding
%data = load('../../data/vortexShedding.mat');
run '../Reconstruction/Cylinder_Re7000/load_data.m'

% %%
% %Calculate vorticity field
% 
% %Initialize a 3D matrix to hold the vorticity for all timesteps
% vorticity_field = zeros(223, 196, 600);
% 
% %Loop through the timesteps, calculate the vorticity, and store it
% for t = 1:600
%     % Extract u and v for this timestep
%     u = data.velocityField(t).u;
%     v = data.velocityField(t).v;
% 
%     % Extract the mesh grids (vMesh_x and vMesh_y)
%     x = data.vMesh_x;
%     y = data.vMesh_y;
% 
%     % Step 3: Compute the partial derivatives using gradient
%     [dudx, dudy] = gradient(u, x(1,:), y(:,1));  % ∂u/∂x and ∂u/∂y
%     [dvdx, dvdy] = gradient(v, x(1,:), y(:,1));  % ∂v/∂x and ∂v/∂y
% 
%     % Step 4: Compute the vorticity: ω = ∂v/∂x - ∂u/∂y
%     vorticity = dvdx - dudy;
% 
%     % Store the vorticity in the 3D matrix
%     vorticity_field(:,:,t) = vorticity;
% end
% 
% % Step 5: Save the 3D vorticity field matrix
% disp(size(vorticity_field))
% 
% vorticity_field_reshaped = reshape(vorticity_field, [223*196, 600]);
% 
% %% Calculate velocity fields
% 
% % Step 1: Initialize matrices to hold the velocity components and magnitude for all timesteps
% num_timesteps = 600;
% u_field = zeros(223, 196, num_timesteps);
% v_field = zeros(223, 196, num_timesteps);
% velocity_magnitude_field = zeros(223, 196, num_timesteps);
% 
% % Step 2: Loop through the timesteps, extract u and v, calculate the magnitude, and store them
% for t = 1:num_timesteps
%     u = data.velocityField(t).u;  % Extract u for timestep t
%     v = data.velocityField(t).v;  % Extract v for timestep t
% 
%     % Store u and v in their respective fields
%     u_field(:,:,t) = u;
%     v_field(:,:,t) = v;
% 
%     % Calculate the velocity magnitude at each point
%     velocity_magnitude_field(:,:,t) = sqrt(u.^2 + v.^2);
% end
% 
% % Step 3: Reshape the fields to 2D matrices for further processing or visualization
% u_field_reshaped = reshape(u_field, [size(u_field, 1) * size(u_field, 2), size(u_field, 3)]);
% v_field_reshaped = reshape(v_field, [size(v_field, 1) * size(v_field, 2), size(v_field, 3)]);
% velocity_magnitude_field_reshaped = reshape(velocity_magnitude_field, [size(velocity_magnitude_field, 1) * size(velocity_magnitude_field, 2), size(velocity_magnitude_field, 3)]);
% 
% % Display sizes of the reshaped fields
% disp(size(u_field_reshaped));
% disp(size(v_field_reshaped));
% disp(size(velocity_magnitude_field_reshaped));

% %% Plotting snapshots
% 
% % Generate high-quality figure
% figure('Units', 'inches', 'Position', [2, 2, 6, 4], 'PaperPositionMode', 'auto');
% 
% % Display the data with enhanced contrast
% data = reshape(u_field_reshaped(:,273), 223, 196);
% imagesc(data), colormap("gray");
% clim([min(data(:)) max(data(:))]); % Set contrast to full data range
% %colorbar;
% 
% % Set axis properties for publication quality
% axis equal tight; % Ensure correct aspect ratio and no white borders
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1, 'XTick', [], 'YTick', []);
% 
% % Title with LaTeX for better formatting
% %title('Snapshot', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
% 
% % Export the figure with high resolution
% %print('illustration_cylinder_wake', '-dpng', '-r300'); % For PNG at 300 DPI
% % print('snapshot_figure', '-depsc', '-r300'); % For EPS vector format

%% Define different matrices

X = velocity_magnitude_field_reshaped(:,:);

X_noiseless = velocity_magnitude_field_reshaped(:,:);

%%

disp('test');
disp(min(X));

%% add salt & pepper noise
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

% %% This takes tttimmmmmeeee
% addpath('..\..\utils')
% [L,S] = ALM_RPCA(X, 1, 1e-5, 1000);
% disp(size(X));
% 
% %% Manual approach for tuning lambda parameter
% 
% [L_1p5, S_1p5] = ALM_RPCA(X, 1.5, 1e-5, 1000);
% disp('----HERE----')
% [L_2, S_2] = ALM_RPCA(X, 2, 1e-5, 1000);
% disp('----HERE----')
% [L_2p5, S_2p5] = ALM_RPCA(X, 2.5, 1e-5, 1000);
% disp('----HERE----')
% [L_3, S_3] = ALM_RPCA(X, 3, 1e-5, 1000);
% disp('----HERE----')
% %%
% % Save all L and S matrices to a .mat file
% save('results_Re7000/RPCA_results.mat', 'X_noiseless', 'X', 'L', 'S', 'L_1p5', 'S_1p5', 'L_2', 'S_2', 'L_2p5', 'S_2p5', 'L_3', 'S_3');
% 
% disp('All L and S matrices have been saved to RPCA_results.mat');

%%

data_Re7000 = load('results_Re7000\RPCA_results.mat');

%%

Re7000_X_orig = data_Re7000.X_noiseless;
Re7000_L = data_Re7000.L;
Re7000_L_1p5 = data_Re7000.L_1p5;
Re7000_L_2 = data_Re7000.L_2;
Re7000_L_2p5 = data_Re7000.L_2p5;
Re7000_L_3 = data_Re7000.L_3;
Re7000_S = data_Re7000.S;
Re7000_X = data_Re7000.X;
Re7000_S_2p5 = data_Re7000.S_2p5;


% %%
% tol=1e-10;
% disp(rank(L, tol))
% disp(rank(X, tol))
% 
% %%
% hfig = figure;
% 
% subplot(2,2,1)
% imagesc(reshape(X_noiseless(:,273),223,196)), colormap gray
% %colorbar;
% %xlabel('$\mathbf{X}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% subplot(2,2,2)
% imagesc(reshape(X(:,273),223,196)), colormap gray
% %colorbar;
% xlabel('$\mathbf{X}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% subplot(2,2,3)
% imagesc(reshape(L(:,273),223,196)), colormap gray
% %colorbar;
% xlabel('$\mathbf{L}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% subplot(2,2,4)
% imagesc(reshape(S(:,273),223,196)), colormap gray
% %colorbar;
% xlabel('$\mathbf{S}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% fname = 'output/RPCA_Re7000_SP10percent_10std_lambda2';
% 
% picturewidth = 20; % set this parameter and keep it forever
% hw_ratio = 1; % feel free to play with this ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
% %legend('Location', 'southwest', 'FontSize', 14, 'FontWeight','bold');
% set(findall(hfig,'-property','Box'),'Box','on') % optional
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
% box on;
% %print(hfig,fname,'-dpdf','-vector','-fillpage')
% %print(hfig,fname,'-dpng','-vector')

%%
% Define the number of colors in the colormap
n = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n)', linspace(0, 1, n)', ones(n, 1); ...
        ones(n, 1), linspace(1, 0, n)', linspace(1, 0, n)'];

%%
colormapeditor;

%%

% Determine the color limits based on X_noiseless
clim = [min(X_noiseless(:))-0.05, max(X_noiseless(:))+0.05];
%clim = [-0.15, 0.15];

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact'); % Adjust spacing

% Plot each component in a separate tile with consistent color limits
nexttile
imagesc(reshape(X_noiseless(:,273),223,196));
colormap(cmap_);
caxis(clim); % Apply color limits
%colorbar;
xlabel('Original')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(X(:,273),223,196)), colormap(cmap_)
caxis(clim); % Apply color limits
colorbar;
xlabel('$\mathbf{X}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(Re7000_L_2p5(:,273),223,196)), colormap(cmap_)
caxis(clim); % Apply color limits
xlabel('$\mathbf{L}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile
imagesc(reshape(Re7000_S_2p5(:,273),223,196)), colormap(cmap_)
caxis(clim); % Apply color limits
%colorbar;
xlabel('$\mathbf{S}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output_Re7000/RPCA_Re7000_SP10percent_10std_2x2_color_lambda2p5';
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


% %%
% 
% % Assuming X, L, and S are your data arrays with dimensions (num_points, num_timesteps)
% num_timesteps = 30; % Number of time steps
% grid_size = [223, 196]; % Dimensions of the grid
% 
% % % Create VideoWriter object
% % filename = 'RPCA_cyl_Re4000_lambda2p1_noise.mp4';  % Output file name
% % v = VideoWriter(filename, 'MPEG-4');
% % v.FrameRate = 10;  % Set frame rate (adjust as needed)
% % 
% % % Open the video writer
% % open(v);
% 
% % Create a figure
% figure;
% set(gcf,'units','points','position',[50,50,800,150]);
% 
% for t = 1:num_timesteps
%     % Reshape and plot X
%     subplot(1, 3, 1);
%     imagesc(reshape(X(:, t), grid_size)); 
%     colormap gray;
%     colorbar;
%     title('Original Image, X');
% 
%     % Reshape and plot L
%     subplot(1, 3, 2);
%     imagesc(reshape(L(:, t), grid_size)); 
%     colormap gray;
%     colorbar;
%     title('Low-Rank Approximation, L');
% 
%     % Reshape and plot S
%     subplot(1, 3, 3);
%     imagesc(reshape(S(:, t), grid_size)); 
%     colormap gray;
%     colorbar;
%     title('Sparse Component, S');
% 
%     % Hide the fourth subplot
%     %subplot(2, 2, 2);
%     axis off;
% 
%     % % Capture the frame
%     % frame = getframe(gcf);
%     % 
%     % % Write the frame to the video file
%     % writeVideo(v, frame);
% 
%     % Pause to create animation effect
%     pause(0.1); % Adjust the pause duration to control animation speed
% end
% 
% % Close the video file
% %close(v);
% 
% %disp(['Animation saved as ' filename]);
% 
% % %% Optimal shrinkage value
% 
% %Finding the size of L
% [n1,n2] = size(L);
% beta = n2/n1; if beta > 1, beta = 1/beta; end
% 
% %Trying to clean L-matrix
% [U_, S_, V_] = svd(L, 'econ');
% y = diag(S_);
% disp(y)
% 
% disp(optimal_shrinkage(beta,0) * median(y));
% limit = optimal_shrinkage(beta,0) * median(y);
% 
% y( y < (optimal_shrinkage(beta,0) * median(y)) ) = 0;
% Lhat = U_ * diag(y) * V_';
% 
% % Plotting
% figure;
% loglog(S_, 'o-', 'DisplayName', 'Cylinder flow', 'MarkerSize', 5); hold on;
% yline(limit, '--', 'LineWidth', 1.5);  % '--' specifies a dotted line, and you can adjust the LineWidth
% 
% %%
% 
% % Assuming X, L, and S are your data arrays with dimensions (num_points, num_timesteps)
% num_timesteps = 30; % Number of time steps
% grid_size = [223, 196]; % Dimensions of the grid
% 
% % Create a figure
% figure;
% set(gcf,'units','points','position',[50,50,800,200]);
% 
% for t = 1:num_timesteps
%     % Reshape and plot X
%     subplot(1, 2, 1);
%     imagesc(reshape(L(:, t), grid_size)); 
%     colormap winter;
%     colorbar;
%     title('Low-Rank Approximation, L');
% 
%     % Reshape and plot L
%     subplot(1, 2, 2);
%     imagesc(reshape(Lhat(:, t), grid_size)); 
%     colormap winter;
%     colorbar;
%     title('Low-Rank Approximation, Lhat, with optimal threshold shrinkage');
% 
%     % Pause to create animation effect
%     pause(0.1); % Adjust the pause duration to control animation speed
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
% %[Gdat_Lhat, f_Lhat, e_r_Lhat] = estpsd(Lhat, N, fs);
% [Gdat_S, f_S, e_r_S] = estpsd(S, N, fs);
% 
% % Average PSD across all spatial points (channels)
% PSD_noiseless = mean(Gdat_noiseless, 2);
% PSD_X = mean(Gdat_X, 2);
% PSD_L = mean(Gdat_L, 2);
% %PSD_Lhat = mean(Gdat_Lhat, 2);
% PSD_S = mean(Gdat_S, 2);
% 
% % Plot the PSDs with publication-quality settings
% figure('Units', 'inches', 'Position', [0, 0, 6, 4]);
% 
% % Plot each dataset with distinct line styles and markers
% loglog(f_noiseless, PSD_noiseless, 'LineWidth', 1.5, 'DisplayName', 'Original noiseless data');
% hold on;
% loglog(f_X, PSD_X, 'LineWidth', 1.5, 'DisplayName', 'X (With noise)');
% loglog(f_L, PSD_L, 'LineWidth', 1.5, 'DisplayName', 'L (Low-Rank)');
% %loglog(f_Lhat, PSD_Lhat, 'k--', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'L hat (G-D thresh)');
% loglog(f_S, PSD_S, 'LineWidth', 1.5, 'DisplayName', 'S (Sparse)');
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
% %print(fullfile(output_dir, 'PSD_cyl_Re7000_SP_noise'), '-dpdf', '-r300');
% % Alternatively, save as EPS if preferred
% % print(fullfile(output_dir, 'psd_plot'), '-depsc2', '-r300');
% 
% 
% 
% %% smoother PSD
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
% loglog(f_X, PSD_X_avg, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'X (Original)');
% hold on;
% loglog(f_noiseless, PSD_noiseless, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Original noiseless data');
% loglog(f_L, PSD_L_avg, 's--', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'L (Low-Rank)');
% loglog(f_S, PSD_S_avg, 'd:', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'S (Sparse)');
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


% %%
% 
% E_k = PSD_L * mean(mean(velocity_magnitude_field_reshaped, 2)) / 2*pi;
% disp(size(E_k))
% 
% k = 2*pi*f_L / mean(mean(velocity_magnitude_field_reshaped, 2));
% disp(size(k))
% 
% loglog(k, E_k)

% %%
% 
% data = u_field; %size(u_field) = [223, 196, 600]
% fs = 15;
% 
% % Extract one row (e.g., the first spatial point) from the data matrix
% velocity = squeeze(data(1, 1, :));
% 
% % Compute the PSD using Welch's method
% [PSD, f] = pwelch(velocity, hann(256), 128, 512, fs);
% 
% % Plot the PSD in decibels
% loglog(f, PSD); xlabel('Frequency'); ylabel('PSD'); title('Power Spectral Density');
% 
% %%
% 
% E_k = PSD * mean(mean(u_field_reshaped, 2)) / 2*pi;
% disp(size(E_k))
% 
% k = 2*pi*f / mean(mean(u_field_reshaped, 2));
% disp(size(k))
% 
% loglog(k, E_k)


% %% PSD 
% 
% % Parameters
% data_orig = X_noiseless; % size(u_field) = [223*196, 600]
% data_noise = X;
% data_L = L;
% data_S = S;
% fs = 15;        % Sampling frequency in Hz
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
% nx = 223;
% ny = 196;
% num_points = nx * ny;
% PSD_orig_ = PSD_orig / num_points;
% PSD_X_ = PSD_X / num_points;
% PSD_L_ = PSD_L / num_points;
% PSD_S_ = PSD_S / num_points;

%%

addpath '..'/utils/

[PSD_orig, f_orig] = PSD_calc(X_noiseless, 15);
[PSD_L, f_L] = PSD_calc(Re7000_L , 15);
%%
[PSD_L_1p5, f_L_1p5] = PSD_calc(Re7000_L_1p5, 15);
[PSD_L_2, f_L_2] = PSD_calc(Re7000_L_2, 15);
[PSD_L_2p5, f_L_2p5] = PSD_calc(Re7000_L_2p5, 15);
[PSD_L_3, f_L_3] = PSD_calc(Re7000_L_3, 15);

%%
[PSD_X, f_X] = PSD_calc(Re7000_X, 15);
%%
[PSD_S, f_S] = PSD_calc(Re7000_S, 15);

%% PLOT X_OG, X, L, S

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'LineWidth', 1.5, 'Displayname', 'Original flow field');
hold on;
loglog(f_X, PSD_X, 'LineWidth', 1.5, 'Displayname', 'X');
loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L');
loglog(f_S, PSD_S, 'LineWidth', 1.5, 'Displayname', 'S');
%hold on;
%loglog(f, f.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
grid on;
%ylim([1e-4 1e5])

fname = 'output_Re7000/PSD_Re7000_RPCA_lambda1';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.55; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

%% PLOT DIFFERENT LAMBDA VALUES

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'k', 'LineWidth', 1.5, 'Displayname', 'Original flow field', 'MarkerSize', 5);
hold on;
loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1$');
loglog(f_L_1p5, PSD_L_1p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1.5$');
loglog(f_L_2, PSD_L_2, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2$');
loglog(f_L_2p5, PSD_L_2p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2.5$');
loglog(f_L_3, PSD_L_3, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 3$');
%hold on;
%loglog(f, f.^(-5/3), 'r-', 'LineWidth', 1.5); % Superimpose the k^-5/3 line
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
grid on;
ylim([1e-10 1])

fname = 'output_Re7000/PSD_Re7000_lambda_tuning';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.55; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

%% Plot and save error

addpath '../utils/'

[res, ssim] = error(X_noiseless, L_3, 0, false);

disp(res);
disp(ssim);

% %% Try plotting side by side nicely
% 
% % Create a tiled layout for two plots side by side
% hfig = figure;
% tiledlayout(1, 2, 'TileSpacing', 'tight', 'Padding', 'tight');
% 
% % First plot
% nexttile;
% loglog(f_orig, PSD_orig, 'LineWidth', 1.5, 'Displayname', 'Original flow field');
% hold on;
% loglog(f_X, PSD_X, 'LineWidth', 1.5, 'Displayname', 'X');
% loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L');
% loglog(f_S, PSD_S, 'LineWidth', 1.5, 'Displayname', 'S');
% xlabel('Frequency (Hz)');
% ylabel('Power Spectral Density');
% legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
% grid on;
% box on;
% 
% % Second plot
% nexttile;
% loglog(f_orig, PSD_orig, 'k', 'LineWidth', 1.5, 'Displayname', 'Original flow field', 'MarkerSize', 5);
% hold on;
% loglog(f_L, PSD_L, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1$');
% loglog(f_L_1p5, PSD_L_1p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1.5$');
% loglog(f_L_2, PSD_L_2, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2$');
% loglog(f_L_2p5, PSD_L_2p5, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 2.5$');
% loglog(f_L_3, PSD_L_3, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 3$');
% xlabel('Frequency (Hz)');
% %ylabel('Power Spectral Density');
% ylim([1e-9 1]);
% set(gca, 'Ytick', []);
% legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
% grid on;
% 
% % Adjust figure properties
% set(findall(hfig,'-property','FontSize'),'FontSize',21); % adjust font size to your document
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
% set(findall(hfig,'-property','Box'),'Box','off') % optional
% picturewidth = 20; % Width in centimeters
% hw_ratio = 0.8; % Height to width ratio
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
% box on;
% 
% % Save as PDF
% fname = 'output_Re7000/PSD_Re7000_combined_tiled';
% %print(hfig, fname, '-dpdf', '-vector'); % Save as PDF
