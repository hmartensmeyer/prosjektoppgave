%clear all, close all, clc
%addpath('../utils');
data = load('../data/CYLINDER_ALL.mat');

%%

% Helpful for plotting
load('../utils/CCcool.mat')
cmap = CC;
clim = [-5, 5];

%%

UALL = data.UALL(:,:);
VALL = data.VALL(:,:);

disp(size(UALL));
disp(size(VALL));

% Calculate the velocity magnitude at each point and time
X = sqrt(UALL.^2 + VALL.^2);  % Shape: (nx, ny, n_timesteps)
X_noiseless = sqrt(UALL.^2 + VALL.^2);

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

imagesc(X(:,:,1));  % Display the first slice of the noisy data
disp(size(X))

%%

% Add the 'utils' folder to MATLAB path
addpath('..\..\utils')  % Adjusted relative path
% Alternatively, use the absolute path:
% addpath('C:\Users\herma\OneDrive\Documents\MATLAB\utils')

% Verify that MATLAB can find the ALM_RPCA function
assert(~isempty(which('ALM_RPCA')), 'ALM_RPCA function not found. Check the path and function name.');

[L,S] = ALM_RPCA(X, 0.3, 1e-5, 1000);

%%
tol=1e-10;
disp(rank(L, tol));
disp(rank(X, tol))
disp(nnz(S));

% %%
% hfig = figure;
% 
% subplot(3,1,1)
% imagesc(reshape(X(:,5),199,449)), colormap gray
% %colorbar;
% xlabel('$\mathbf{X}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% subplot(3,1,2)
% imagesc(reshape(L(:,5),199,449)), colormap gray
% %colorbar;
% xlabel('$\mathbf{L}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% subplot(3,1,3)
% imagesc(reshape(S(:,5),199,449)), colormap gray
% %colorbar;
% xlabel('$\mathbf{S}$')
% set(gca, 'XTick', [], 'YTick', []) % Remove axis ticks
% %axis off;
% 
% fname = 'output/RPCA_Re100_SP10percent_10std';
% 
% picturewidth = 20; % set this parameter and keep it forever
% hw_ratio = 1.0; % feel free to play with this ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
% %legend('Location', 'southwest', 'FontSize', 14, 'FontWeight','bold');
% set(findall(hfig,'-property','Box'),'Box','on') % optional
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
% box on;
% print(hfig,fname,'-dpdf','-vector','-fillpage')
% print(hfig,fname,'-dpng','-vector')

%%
% Define the number of colors in the colormap
n = 256;

% Create a blue to white to red colormap
cmap_ = [linspace(0, 1, n)', linspace(0, 1, n)', ones(n, 1); ...
        ones(n, 1), linspace(1, 0, n)', linspace(1, 0, n)'];

%%

% Set up the figure and tiled layout
hfig = figure;
tiledlayout(2 ,2, 'TileSpacing', 'compact', 'Padding', 'tight'); % Adjust spacing

clim = [min(X_noiseless(:)-0.4), max(X_noiseless(:)+0.4)];
%clim = ([0, 2]);

% Plot each component in a separate tile
nexttile(1)
imagesc(reshape(X_noiseless(:,5),199,449)), colormap(cmap_)
xlabel('Original')
caxis(clim);
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(2)
imagesc(reshape(X(:,5),199,449)), colormap(cmap_)
xlabel('$\mathbf{X}$')
caxis(clim);
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(3)
imagesc(reshape(L(:,5),199,449)), colormap(cmap_)
xlabel('$\mathbf{L}$')
caxis(clim);
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

nexttile(4)
imagesc(reshape(S(:,5),199,449)), colormap(cmap_)
xlabel('$\mathbf{S}$')
colorbar;
caxis(clim);
set(gca, 'XTick', [], 'YTick', []); % Remove axis ticks

% Set additional properties
fname = 'output_Re100/RPCA_Re100_SP10percent_10std';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.6; % Height-width ratio

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

% Assuming X, L, and S are your data arrays with dimensions (num_points, num_timesteps)
num_timesteps = 50; % Number of time steps
grid_size = [199, 449]; % Dimensions of the grid

% Create a figure
figure;

for t = 1:num_timesteps
    % Reshape and plot X
    subplot(2, 2, 1);
    imagesc(reshape(X(:, t), grid_size)); 
    colormap winter;
    colorbar;
    title('Original Image, X');
    
    % Reshape and plot L
    subplot(2, 2, 3);
    imagesc(reshape(L(:, t), grid_size)); 
    colormap winter;
    colorbar;
    title('Low-Rank Approximation, L');
    
    % Reshape and plot S
    subplot(2, 2, 4);
    imagesc(reshape(S(:, t), grid_size)); 
    colormap winter;
    colorbar;
    title('Sparse Component, S');
    
    % Hide the fourth subplot
    subplot(2, 2, 2);
    axis off;
    
    % Pause to create animation effect
    pause(0.02); % Adjust the pause duration to control animation speed
end

%% Plot and save error

addpath '../utils/'

[res, ssim] = error(X_noiseless, L, 0, false);