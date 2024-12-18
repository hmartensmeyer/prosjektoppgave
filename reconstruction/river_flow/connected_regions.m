close all
format short;
%% SVD over data to remove color gradient
[U_og, S_og, V_og] = svd(data_normalized(:, 1:1050), 'econ');
[U_rec_3, S_rec_3, V_rec_3] = svd(recon_og_clu{3}(:, :), 'econ');
[U_rec_11, S_rec_11, V_rec_11] = svd(recon_og_clu{11}(:, :), 'econ');
[U_rec_31, S_rec_31, V_rec_31] = svd(recon_og_clu{31}(:, :), 'econ');

data_og = U_og(:, 2:end) * S_og(2:end, 2:end) * V_og(:, 2:end)';
data_rec_3 = U_rec_3(:, 2:end) * S_rec_3(2:end, 2:end) * V_rec_3(:, 2:end)';
data_rec_11 = U_rec_11(:, 2:end) * S_rec_11(2:end, 2:end) * V_rec_11(:, 2:end)';
data_rec_31 = U_rec_31(:, 2:end) * S_rec_31(2:end, 2:end) * V_rec_31(:, 2:end)';
%%

timestep = 886; %example timestep

X_og = reshape(data_og(:, timestep), nx, ny);
X_rec_3 = reshape(data_rec_3(:, timestep), nx, ny);
X_rec_11 = reshape(data_rec_11(:, timestep), nx, ny);
X_rec_31 = reshape(data_rec_31(:, timestep), nx, ny);

% Normalize each dataset to the range [0, 1]
X_og = (X_og - min(X_og(:))) / (max(X_og(:)) - min(X_og(:)));
X_rec_3 = (X_rec_3 - min(X_rec_3(:))) / (max(X_rec_3(:)) - min(X_rec_3(:)));
X_rec_11 = (X_rec_11 - min(X_rec_11(:))) / (max(X_rec_11(:)) - min(X_rec_11(:)));
X_rec_31 = (X_rec_31 - min(X_rec_31(:))) / (max(X_rec_31(:)) - min(X_rec_31(:)));

%% %%%%%%%%%% CONNECTED AREAS ANALYSIS %%%%%%%%%%%%%%

%Set threshhold between 0 (black) and 255 (white)
thresh = 0.65;
invthresh = 0.35;
areastokeep = 20;

% %%
% %Make binary images
% X1w = X1*0 + (X1>thresh);       %For light-area
% X2w = X2*0 + (X2>thresh);
% 
% CC1w = bwconncomp(X1w);
% CC2w = bwconncomp(X2w);
% 
% areas1w = cell2mat(struct2cell(regionprops(CC1w,"Area")));
% areas2w = cell2mat(struct2cell(regionprops(CC2w,"Area")));
% 
% %Lists of sizes of areas
% [Asz1,order1] = sort(areas1w,'descend');
% [Asz2,order2] = sort(areas2w,'descend');
% n1 = length(order1); n2 = length(order2);
% 
% 
% figure
% semilogy(1:length(Asz1),Asz1,1:length(Asz2),Asz2)
% legend('No waves', 'With waves');
% ylim([5 max([Asz1(1) Asz2(1)])])
% title('Area sizes, hot areas areas');
% ylabel('Area (pixels)');
% xlabel('Area index');
% 
% 
% %Separate out the largest areas
% bigidx1 = order1(1:areastokeep); 
% bigidx2 = order2(1:areastokeep);
% smallidx1 = order1(areastokeep+1:n1); 
% smallidx2 = order2(areastokeep+1:n2); 
% 
% X1tBig = cc2bw(CC1w,ObjectsToKeep=bigidx1);
% X2tBig = cc2bw(CC2w,ObjectsToKeep=bigidx2);
% X1tSmall = cc2bw(CC1w,ObjectsToKeep=smallidx1);
% X2tSmall = cc2bw(CC2w,ObjectsToKeep=smallidx2);
% 
% f=figure;
% set(f,"Position",[ 133         549        1594         730])
% colormap gray
% 
% t=tiledlayout(4,2,'Padding','compact','TileSpacing','compact');
% nexttile, imagesc(X1), axis off,title('Original, no waves');
% nexttile, imagesc(X2), axis off, title('Original, with waves');
% nexttile, imagesc(X1w), axis off,title(sprintf('No waves. Threshold: %d',thresh));
% nexttile, imagesc(X2w), axis off, title(sprintf('With waves. Threshold: %d',thresh));
% nexttile, imagesc(X1tBig), axis off, title(sprintf('Threshold: %d, Only %d largest warm areas',thresh,areastokeep));
% nexttile, imagesc(X2tBig), axis off, title(sprintf('Threshold: %d, Only %d largest warm areas',thresh,areastokeep));
% nexttile, imagesc(X1tSmall), axis off, title(sprintf('Threshold: %d, Without %d largest warm areas',thresh,areastokeep));
% nexttile, imagesc(X2tSmall), axis off, title(sprintf('Threshold: %d, Without %d largest warm areas',thresh,areastokeep));
% 

%% %%%%%%%%%%%% SAME FOR DARK AREAS %%%%%%%%%%%%%% 
X_ogb = X_og*0 + (X_og<invthresh) + (X_og>thresh);    %For dark-area analysis
X_rec_3b = X_rec_3*0 + (X_rec_3<invthresh) + (X_rec_3>thresh);
X_rec_11b = X_rec_11*0 + (X_rec_11<invthresh) + (X_rec_11>thresh);
X_rec_31b = X_rec_31*0 + (X_rec_31<invthresh) + (X_rec_31>thresh);
%S_rec_11b = S_rec_11*0 + (S_rec_11<invthresh);

hfig = figure;
colormap gray;
tiledlayout(2, 2, Padding="compact", TileSpacing="compact");

nexttile; imagesc(X_og); colorbar;

nexttile; imagesc(X_rec_11); colorbar;

nexttile; imagesc(1-X_ogb); colorbar;

nexttile; imagesc(1-X_rec_11b); colorbar;

%%

CC_ogb = bwconncomp(X_ogb);
CC_rec_3b = bwconncomp(X_rec_3b);
CC_rec_11b = bwconncomp(X_rec_11b);
CC_rec_31b = bwconncomp(X_rec_31b);
%CC_S_rec_11b = bwconncomp(S_rec_11b);

areas_ogb = cell2mat(struct2cell(regionprops(CC_ogb,"Area")));
areas_rec_3b = cell2mat(struct2cell(regionprops(CC_rec_3b,"Area")));
areas_rec_11b = cell2mat(struct2cell(regionprops(CC_rec_11b,"Area")));
areas_rec_31b = cell2mat(struct2cell(regionprops(CC_rec_31b,"Area")));
%areas_S_rec_11b = cell2mat(struct2cell(regionprops(CC_S_rec_11b,"Area")));

%Lists of sizes of areas
[Asz_og,order_og] = sort(areas_ogb,'descend');
[Asz_rec_3,order_rec_3] = sort(areas_rec_3b,'descend');
[Asz_rec_11,order_rec_11] = sort(areas_rec_11b,'descend');
[Asz_rec_31,order_rec_31] = sort(areas_rec_31b,'descend');
%[Asz_S_rec_11,order_S_rec_11] = sort(areas_S_rec_11b,'descend');

% figure
% semilogy(1:length(Asz_og),Asz_og,1:length(Asz_rec_11),Asz_rec_11)
% legend('Original', 'Reconstruction');
% ylim([5 max([Asz_og(1) Asz_rec_11(1)])])
% title('Area sizes, darkest areas');
% ylabel('Area (pixels)');
% xlabel('Area index');

% Plot area sizes for all four datasets
figure
loglog(1:length(Asz_og), Asz_og, ...
         1:length(Asz_rec_11), Asz_rec_11, ...
         1:length(Asz_rec_31), Asz_rec_31, 'LineWidth', 1.5)
legend('Original', 'Reconstruction, 11 sensors', 'Reconstruction, 31 sensors', 'Location', 'best');
ylim([5 max([Asz_og(1), Asz_rec_3(1), Asz_rec_11(1), Asz_rec_31(1)])])
title('Area sizes, darkest areas');
ylabel('Area (pixels)');
xlabel('Area index');
grid on


%Just the largest areas
bigidx_og = order_og(1:areastokeep); 
bigidx_rec_3 = order_rec_3(1:areastokeep);
bigidx_rec_11 = order_rec_11(1:areastokeep);
bigidx_rec_31 = order_rec_31(1:areastokeep);
%bigidx_S_rec_11 = order_S_rec_11(1:areastokeep);

X_og_tBig = cc2bw(CC_ogb, ObjectsToKeep=bigidx_og);
X_rec_3_tBig = cc2bw(CC_rec_3b,ObjectsToKeep=bigidx_rec_3);
X_rec_11_tBig = cc2bw(CC_rec_11b,ObjectsToKeep=bigidx_rec_11);
X_rec_31_tBig = cc2bw(CC_rec_31b,ObjectsToKeep=bigidx_rec_31);
%S_rec_11_tBig = cc2bw(CC_S_rec_11b,ObjectsToKeep=bigidx_S_rec_11);

%%
hfig = figure;
%set(f,"Position",[ 183         200        900         800])
colormap gray
 
tiledlayout(3,3,'Padding','compact','TileSpacing', 'compact');

nexttile, imshow(X_og), axis off,title('Original');
nexttile, imshow(X_rec_11), axis off, title('11 sensors');
nexttile, imshow(X_rec_31), axis off, title('31 sensors');
nexttile, imshow(X_ogb), axis off;
nexttile, imshow(X_rec_11b), axis off;%, title(sprintf('Reconstruction, 11 sensors. Threshold: %d',invthresh));
nexttile, imshow(X_rec_31b), axis off;%, title(sprintf('Reconstruction, 31 sensors. Threshold: %d',invthresh));
nexttile, imshow(X_og_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));
nexttile, imshow(X_rec_11_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));
nexttile, imshow(X_rec_31_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));

ax = findall(hfig, 'Type', 'axes');
set(ax, 'Box', 'on', 'XTick', [], 'YTick', []);

% Set additional properties
fname = 'output_Nidelva/comparing_areas_3x3';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 1; % Height-width ratio

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


%% Try to calculate the dark and light area coverage for every timestep and compare reconstructions

% Preallocate arrays to store the number of pixels below invthresh for each timestep
num_pixels_ogb = zeros(1050, 1);
num_pixels_rec_11b = zeros(1050, 1);
num_pixels_rec_31b = zeros(1050, 1);

% Loop through all timesteps
for timestep = 1:1050
    disp(num2str(timestep));
    % Reshape data for the current timestep
    X_og = reshape(data_og(:, timestep), nx, ny);
    X_rec_3 = reshape(data_rec_3(:, timestep), nx, ny);
    X_rec_11 = reshape(data_rec_11(:, timestep), nx, ny);
    X_rec_31 = reshape(data_rec_31(:, timestep), nx, ny);

    % Normalize the data to [0, 1]
    X_og = (X_og - min(X_og(:))) / (max(X_og(:)) - min(X_og(:)));
    X_rec_3 = (X_rec_3 - min(X_rec_3(:))) / (max(X_rec_3(:)) - min(X_rec_3(:)));
    X_rec_11 = (X_rec_11 - min(X_rec_11(:))) / (max(X_rec_11(:)) - min(X_rec_11(:)));
    X_rec_31 = (X_rec_31 - min(X_rec_31(:))) / (max(X_rec_31(:)) - min(X_rec_31(:)));

    % Create binary images for pixels below invthresh
    X_ogb = X_og < invthresh;
    X_rec_3b = X_rec_3 < invthresh;
    X_rec_11b = X_rec_11 < invthresh;
    X_rec_31b = X_rec_31 < invthresh;

    % Count the number of pixels below invthresh
    num_pixels_ogb(timestep) = sum(X_ogb(:));
    num_pixels_rec_3b(timestep) = sum(X_rec_3b(:));
    num_pixels_rec_11b(timestep) = sum(X_rec_11b(:));
    num_pixels_rec_31b(timestep) = sum(X_rec_31b(:));
end

%% Plot the results
figure;
plot(1:1050, num_pixels_rec_3b, 'DisplayName', 'Reconstruction 3', 'LineWidth', 1.5); hold on;
plot(1:1050, num_pixels_rec_11b, 'DisplayName', 'Reconstruction 11', 'LineWidth', 1.5); hold on;
plot(1:1050, num_pixels_rec_31b, 'DisplayName', 'Reconstruction 31', 'LineWidth', 1.5);
plot(1:1050, num_pixels_ogb, 'DisplayName', 'Original', 'LineWidth', 1.5);
legend('Location', 'best');
xlabel('Timestep');
ylabel('Number of Pixels Below invthresh');
title('Pixels Below Threshold Over Time');
grid on;
hold off;

%% Define the window size for the moving average
window_size = 20;

% Apply moving average to smooth the data
smoothed_pixels_ogb = movmean(num_pixels_ogb, window_size);
smoothed_pixels_rec_3b = movmean(num_pixels_rec_3b, window_size);
smoothed_pixels_rec_11b = movmean(num_pixels_rec_11b, window_size);
smoothed_pixels_rec_31b = movmean(num_pixels_rec_31b, window_size);

% Plot the smoothed results
hfig = figure;
%plot(1:1050, smoothed_pixels_ogb, 'DisplayName', 'Original', 'LineWidth', 1.5); hold on;
%semilogy(1:1050, smoothed_pixels_rec_3b, 'DisplayName', 'Reconstruction 3', 'LineWidth', 1.5); hold on;
semilogy(1:1050, smoothed_pixels_rec_11b, 'DisplayName', '11 sensors', 'LineWidth', 1.5); hold on;
semilogy(1:1050, smoothed_pixels_rec_31b, 'DisplayName', '31 sensors', 'LineWidth', 1.5);
semilogy(1:1050, smoothed_pixels_ogb, 'k', 'DisplayName', 'Original', 'LineWidth', 1.5);
xlabel('Timestep');
ylabel('Area (pixels)');
xlim([0,1100])
%set(gca, 'XTick', [], 'YTick', [])
grid on;

% Set additional properties
fname = 'output_Nidelva/darklight_pixel_coverage_US10';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.40; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location', 'southeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;

% Export the figure
print(hfig, fname, '-dpdf');%, '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-vector');


% %%
% 
% figure('Position',[100, 100, 1300, 400]);
% t = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
% nexttile;
% imagesc(1-X_ogb); colormap gray;
% nexttile;
% imagesc(1-X_rec_11b); colormap gray;
% nexttile;
% imagesc(1-X_rec_31b); colormap gray;



%% Calculate number of pixels below invtresh for each case
num_pixels_ogb = sum(X_ogb(:));
num_pixels_rec_3b = sum(X_rec_3b(:));
num_pixels_rec_11b = sum(X_rec_11b(:));
num_pixels_rec_31b = sum(X_rec_31b(:));

% Display the results
fprintf('Number of pixels below invtresh:\n');
fprintf('Original: %d\n', num_pixels_ogb);
fprintf('Reconstruction 3: %d\n', num_pixels_rec_3b);
fprintf('Reconstruction 11: %d\n', num_pixels_rec_11b);
fprintf('Reconstruction 31: %d\n', num_pixels_rec_31b);



% %% Time Series Connected Areas Analysis
% 
% close all;
% 
% % Assuming data_reconstructed and recon_svd_clu are 3D arrays: [nx, ny, nt]
% % nt is the number of time steps
% % Replace nx, ny, and nt with your actual dimensions
% % Example initialization (replace with your actual data):
% X1_ = reshape(data_og_nocol(:, 1:50), nx, ny, 50);
% X2_ = reshape(data_rec_nocol(:, 1:50), nx, ny, 50);
% % Normalize X1 and X2 to the range [0, 1]
% for t = 1:size(X1_, 3)
%     X1_(:, :, t) = (X1_(:, :, t) - min(X1_(:, :, t), [], 'all')) / ...
%                    (max(X1_(:, :, t), [], 'all') - min(X1_(:, :, t), [], 'all'));
%     X2_(:, :, t) = (X2_(:, :, t) - min(X2_(:, :, t), [], 'all')) / ...
%                    (max(X2_(:, :, t), [], 'all') - min(X2_(:, :, t), [], 'all'));
% end
% 
% 
% 
% % Parameters
% thresh = 0.5;
% invthresh = 0.45;
% areastokeep = 20;
% 
% % Get the number of time steps
% nt = 10;
% 
% % Loop over each time step
% for t = 12:15
%     % Extract images at time t
%     X1 = X1_(:, :, t);
%     X2 = X2_(:, :, t);
% 
%     %%%%%%%%%%%%% SAME FOR DARK AREAS %%%%%%%%%%%%%%
%     X1b = X1*0 + (X1<invthresh);    %For dark-area analysis
%     X2b = X2*0 + (X2<invthresh);
% 
%     figure;
%     imagesc(X1);
%     colorbar;
%     colormap gray;
%     title(sprintf('X1 at time %d', t));
% 
%     figure;
%     imagesc(X1b);
%     colorbar;
%     colormap gray;
%     title(sprintf('X1b at time %d', t));
% 
%     CC1b = bwconncomp(X1b);
%     CC2b = bwconncomp(X2b);
% 
%     areas1b = cell2mat(struct2cell(regionprops(CC1b, "Area")));
%     areas2b = cell2mat(struct2cell(regionprops(CC2b, "Area")));
% 
%     % Lists of sizes of areas
%     [Asz1b, order1b] = sort(areas1b, 'descend');
%     [Asz2b, order2b] = sort(areas2b, 'descend');
% 
%     % Plot area sizes for dark areas
%     figure;
%     semilogy(1:length(Asz1b), Asz1b, 1:length(Asz2b), Asz2b);
%     legend('No waves', 'With waves');
%     ylim([5, max([Asz1b(1), Asz2b(1)])]);
%     title(sprintf('Area sizes, cold areas, time step %d', t));
%     ylabel('Area (pixels)');
%     xlabel('Area index');
% 
%     % Just the largest areas
%     bigidx1b = order1b(1:min(areastokeep, length(order1b))); 
%     bigidx2b = order2b(1:min(areastokeep, length(order2b)));
%     X1tBigb = ismember(labelmatrix(CC1b), bigidx1b);
%     X2tBigb = ismember(labelmatrix(CC2b), bigidx2b);
% 
%     f = figure;
%     set(f, "Position", [183, 200, 700, 800]);
%     colormap gray;
% 
%     t2 = tiledlayout(3, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%     nexttile, imagesc(X1), axis off, title(sprintf('Original, no waves, time %d', t)); colorbar;
%     nexttile, imagesc(X2), axis off, title(sprintf('Original, with waves, time %d', t)); colorbar;
%     nexttile, imagesc(1 - X1b), axis off, title(sprintf('No waves. Threshold: %.2f', invthresh));
%     nexttile, imagesc(1 - X2b), axis off, title(sprintf('With waves. Threshold: %.2f', invthresh));
%     nexttile, imagesc(1 - X1tBigb), axis off, title(sprintf('Threshold: %.2f, Only %d largest cold areas', invthresh, areastokeep));
%     nexttile, imagesc(1 - X2tBigb), axis off, title(sprintf('Threshold: %.2f, Only %d largest cold areas', invthresh, areastokeep));
% 
% end
% 


% %% RPCA test
% 
% addpath ../../utils/
% 
% [L_rec_11, S_rec_11] = ALM_RPCA(X_rec_11, 0.5, 1e-5, 1000);
% S_rec_11 = (S_rec_11 - min(S_rec_11(:))) / (max(S_rec_11(:)) - min(S_rec_11(:))); %Normalize
% 
% figure;
% t = tiledlayout(2,2,Padding="compact",TileSpacing="compact");
% 
% nexttile(1);
% imagesc(X_og);
% colormap gray;
% 
% nexttile(3);
% imagesc(L_rec_11);
% colormap gray;
% 
% nexttile(4);
% imagesc(S_rec_11);
% colormap gray;