run '../Reconstruction/Nidelva/load_data.m'

%% Define different matrices

X = data_normalized;

% %% add salt & pepper noise
% eta = 0.1; % to add noise change eta to fraction of occluded points. eta = 0.2 for 20%
% if eta ~=0
%     rep = std(X(:))*10;
%     x = rand(size(X(1:end,:)));
%     b = sort(x(:));
%     thresh = b(floor(.5*eta*numel(b)));
% 
%     X(x<thresh) = rep;
% 
%     x = rand(size(X(1:end,:)));
%     b = sort(x(:));
%     thresh = b(floor(.5*eta*numel(b)));
% 
%     X(x<thresh) = -rep;
% end

%% This takes tttimmmmmeeee
addpath('..\..\utils')
[L_1,S_1] = ALM_RPCA(X(:, 1:100), 1, 1e-5, 1000);
disp('Forste ferdig')
[L_15,S_15] = ALM_RPCA(X(:, 1:100), 1.5, 1e-5, 1000);
disp('Andre ferdig')
[L_2, S_2] = ALM_RPCA(X(:, 1:100), 2, 1e-5, 1000);
disp('Tredje ferdig')
[L_25,S_25] = ALM_RPCA(X(:, 1:100), 2.5, 1e-5, 1000);
disp('Fjerde ferdig')
[L_3,S_3] = ALM_RPCA(X(:, 1:100), 3, 1e-5, 1000);
disp('Femte ferdig')
[L_03,S_03] = ALM_RPCA(X(:, 1:100), 0.3, 1e-5, 1000);
disp('Sjette ferdig')



%%
timestep = 95;

% Determine the color limits based on X_noiseless
%clim = [min(X(:))-0.05, max(X(:))+0.05];
%clim = [0,1]

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(3,3, 'TileSpacing', 'compact', 'Padding', 'compact'); % Adjust spacing

% Plot each component in a separate tile with consistent color limits
nexttile(1)
imagesc(reshape(X(:,timestep),nx,ny));
colormap(gray);
%caxis(clim); % Apply color limits
%colorbar;
title('Original')
ylabel('$\kappa = 0.3$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(2)
imagesc(reshape(L_03(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
title('$\mathbf{L}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(3)
imagesc(reshape(S_03(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
%colorbar;
title('$\mathbf{S}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(4)
imagesc(reshape(X(:,timestep),nx,ny));
colormap(gray);
%caxis(clim); % Apply color limits
%colorbar;
%xlabel('Original')
ylabel('$\kappa = 1$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(5)
imagesc(reshape(L_1(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
%xlabel('$\mathbf{L}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(6)
imagesc(reshape(S_1(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
%colorbar;
%xlabel('$\mathbf{S}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(7)
imagesc(reshape(X(:,timestep),nx,ny));
colormap(gray);
%caxis(clim); % Apply color limits
%colorbar;
%xlabel('Original')
ylabel('$\kappa = 3$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(8)
imagesc(reshape(L_3(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
%xlabel('$\mathbf{L}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(9)
imagesc(reshape(S_3(:,timestep),nx,ny)), colormap(gray)
%caxis(clim); % Apply color limits
%colorbar;
%xlabel('$\mathbf{S}$')
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output_Nidelva/RPCA_Nidelva_lambda_tuning';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 1; % Height-width ratio

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
% Define the number of colors in the colormap
n = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n)', linspace(0, 1, n)', ones(n, 1); ...
        ones(n, 1), linspace(1, 0, n)', linspace(1, 0, n)'];

%% Estimate PSD for the different cases (whats the point?)

addpath '..'/utils/

[PSD_orig, f_orig] = PSD_calc(X, 24);
[PSD_L_1, f_L_1] = PSD_calc(L_1, 24);
[PSD_L_15, f_L_15] = PSD_calc(L_15, 24);
[PSD_L_2, f_L_2] = PSD_calc(L_2, 24);
[PSD_L_25, f_L_25] = PSD_calc(L_25, 24);
[PSD_L_3, f_L_3] = PSD_calc(L_3, 24);
[PSD_L_03, f_L_03] = PSD_calc(L_03, 24);


%% PLOT DIFFERENT LAMBDA VALUES

% Plot the Average PSD in decibels
hfig = figure;
loglog(f_orig, PSD_orig, 'k', 'LineWidth', 1.5, 'Displayname', 'Original flow field', 'MarkerSize', 5);
hold on;
loglog(f_L_1, PSD_L_1, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1$');
loglog(f_L_15, PSD_L_15, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 1.5$');
loglog(f_L_3, PSD_L_3, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 3$');
loglog(f_L_03, PSD_L_03, 'LineWidth', 1.5, 'Displayname', 'L, $\kappa = 0.3$');
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
%print(hfig,fname,'-dpdf','-vector');%,'-bestfit')
%print(hfig,fname,'-dpng','-vector')

%%

[res, ssim] = error_calc(X(:, 1:100), L_3, 0, false);

disp(res);
disp(ssim);