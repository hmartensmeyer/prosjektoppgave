nidelva_us10 = load('results_Nidelva\error_sensor_loop.mat');
nidelva_us5 = load('results_Nidelva\error_sensor_loop_US5.mat');

%%
nidelva_us10_meansub = load('results_Nidelva\error_sensor_loop_US10_meansub.mat');
nidelva_us5_meansub = load('results_Nidelva\error_sensor_loop_US5_meansub.mat');

%%

run '..'/Nidelva/load_data.m;

%% Define params

recon_us10 = nidelva_us10.recon_og_clu;
res_us10 = nidelva_us10.res_og_clu;
ssim_us10 = nidelva_us10.ssim_og_clu;

recon_us5 = nidelva_us5.recon_og_clu;
res_us5 = nidelva_us5.res_og_clu;
ssim_us5 = nidelva_us5.ssim_og_clu;

recon_us10_meansub = nidelva_us10_meansub.recon_og_clu;
res_us10_meansub = nidelva_us10_meansub.res_og_clu;
ssim_us10_meansub = nidelva_us10_meansub.ssim_og_clu;

recon_us5_meansub = nidelva_us5_meansub.recon_og_clu;
res_us5_meansub = nidelva_us5_meansub.res_og_clu;
ssim_us5_meansub = nidelva_us5_meansub.ssim_og_clu;

%% Plot SSIM vs Number of Sensors for both cases
hfig = figure;
mean_ssim_us10 = mean(ssim_us10, 2); % Average SSIM over timesteps
mean_ssim_us5 = mean(ssim_us5, 2);
mean_ssim_us10_meansub = mean(ssim_us10_meansub, 2); % Average SSIM over timesteps
mean_ssim_us5_meansub = mean(ssim_us5_meansub, 2);
plot(1:2:31, mean_ssim_us5(1:2:end), 'ko-', 'LineWidth',1.5, 'DisplayName', 'SF = 5'); hold on;
plot(1:2:31, mean_ssim_us5_meansub(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 5 (mean subtracted)');
plot(1:2:31, mean_ssim_us10(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 10');
plot(1:2:31, mean_ssim_us10_meansub(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 10 (mean subtracted)');
xlabel('Sensors');
ylabel('SSIM');
xlim([1,32])
grid on;

% Set additional properties
fname = 'output_Nidelva/ssim_sensortuning_meansub';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','southeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);

% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;

% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% Example: Plot MSE vs Number of Sensors

hfig = figure;
mean_res_us10 = mean(res_us10, 2); % Average SSIM over timesteps
mean_res_us5 = mean(res_us5, 2);
mean_res_us10_meansub = mean(res_us10_meansub, 2); % Average SSIM over timesteps
mean_res_us5_meansub = mean(res_us5_meansub, 2);
plot(1:2:31, mean_res_us5(1:2:end), 'ko-', 'LineWidth',1.5, 'DisplayName', 'SF = 5'); hold on;
plot(1:2:31, mean_res_us5_meansub(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 5 (mean subtracted)');
plot(1:2:31, mean_res_us10(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 10');
plot(1:2:31, mean_res_us10_meansub(1:2:end), 'o-', 'LineWidth',1.5, 'DisplayName', 'SF = 10 (mean subtracted)');
xlabel('Sensors');
ylabel('NRMSR');
grid on;
xlim([1 32]);
ylim([0, 0.25])

% Set additional properties
fname = 'output_Nidelva/mse_sensortuning_meansub';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio

set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','northeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
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

%% Plot example snapshots for one case only?

hfig = figure;
tiledlayout(2, 2, "TileSpacing","compact",'Padding','compact');
timestep = 886; %% 886 for figs, 673 shows 11 underperforming

nexttile;
imagesc(reshape(data_normalized(:, timestep) - mean(data_normalized, 2), nx, ny))
colormap gray;
xlabel('Original')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_us10{3}(:, timestep) - mean(data_normalized, 2), nx, ny))
colormap gray;
xlabel('Reconstructed, 3 sensors')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_us10{13}(:, timestep) - mean(data_normalized, 2), nx, ny))
colormap gray;
xlabel('Reconstructed, 13 sensors')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_us10{31}(:, timestep) - mean(data_normalized, 2), nx, ny))
colormap gray;
xlabel('Reconstructed, 31 sensors')
set(gca, 'XTick', [], 'YTick', [])

% Set additional properties
fname = 'output_Nidelva/og_vs_recon_3_13_31_meansub_after_recon';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.9; % Height-width ratio

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


%% Comparing us10 and us5 for one snapshot

hfig = figure;
tiledlayout(1, 3, "TileSpacing","compact",'Padding','compact');
timestep = 333; %% 886 for figs, 673 shows 11 underperforming

nexttile;
imagesc(reshape(data_normalized(:, timestep), nx, ny))
colormap gray;
xlabel('Original')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_us5{13}(:, timestep), nx, ny))
colormap gray;
xlabel('Reconstruction')
set(gca, 'XTick', [], 'YTick', [])

nexttile;
imagesc(reshape(recon_us10{13}(:, timestep), nx, ny))
colormap gray;
xlabel('Reconstruction')
set(gca, 'XTick', [], 'YTick', [])

% Set additional properties
fname = 'output_Nidelva/comparing_og_us5_us10';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.35; % Height-width ratio

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

% %% Connected region analysis
% 
% % %% SVD over data to remove color gradient
% % [U_og, S_og, V_og] = svd(data_normalized(:, 1:1050), 'econ');
% % %[U_rec_3_us10, S_rec_3_us10, V_rec_3_us10] = svd(recon_us10{3}(:, :), 'econ');
% % [U_rec_13_us10, S_rec_13_us10, V_rec_13_us10] = svd(recon_us10{13}(:, :), 'econ');
% % [U_rec_31_us10, S_rec_31_us10, V_rec_31_us10] = svd(recon_us10{31}(:, :), 'econ');
% % %[U_rec_3_us5, S_rec_3_us5, V_rec_3_us5] = svd(recon_us5{3}(:, :), 'econ');
% % [U_rec_13_us5, S_rec_13_us5, V_rec_13_us5] = svd(recon_us5{13}(:, :), 'econ');
% % [U_rec_31_us5, S_rec_31_us5, V_rec_31_us5] = svd(recon_us5{31}(:, :), 'econ');
% % 
% % %% reconstruct original
% % data_og = U_og(:, 2:end) * S_og(2:end, 2:end) * V_og(:, 2:end)';
% % % reconstruct us10
% % %data_rec_3_us10 = U_rec_3_us10(:, 2:end) * S_rec_3_us10(2:end, 2:end) * V_rec_3_us10(:, 2:end)';
% % data_rec_13_us10 = U_rec_13_us10(:, 2:end) * S_rec_13_us10(2:end, 2:end) * V_rec_13_us10(:, 2:end)';
% % data_rec_31_us10 = U_rec_31_us10(:, 2:end) * S_rec_31_us10(2:end, 2:end) * V_rec_31_us10(:, 2:end)';
% % % reconstruct us 5
% % %data_rec_3_us5 = U_rec_3_us5(:, 2:end) * S_rec_3_us5(2:end, 2:end) * V_rec_3_us5(:, 2:end)';
% % data_rec_13_us5 = U_rec_13_us5(:, 2:end) * S_rec_13_us5(2:end, 2:end) * V_rec_13_us5(:, 2:end)';
% % data_rec_31_us5 = U_rec_31_us5(:, 2:end) * S_rec_31_us5(2:end, 2:end) * V_rec_31_us5(:, 2:end)';
% 
% %%
% timestep = 888;
% 
% % Reshape data to grid
% X_og = reshape(data_og(:, timestep), nx, ny);
% %X_rec_3_us10 = reshape(data_rec_3_us10(:, timestep), nx, ny);
% X_rec_13_us10 = reshape(data_rec_13_us10(:, timestep), nx, ny);
% X_rec_31_us10 = reshape(data_rec_31_us10(:, timestep), nx, ny);
% %X_rec_3_us5 = reshape(data_rec_3_us5(:, timestep), nx, ny);
% X_rec_13_us5 = reshape(data_rec_13_us5(:, timestep), nx, ny);
% X_rec_31_us5 = reshape(data_rec_31_us5(:, timestep), nx, ny);
% 
% % Normalize each dataset to the range [0, 1]
% X_og = (X_og - min(X_og(:))) / (max(X_og(:)) - min(X_og(:)));
% %X_rec_3_us10 = (X_rec_3_us10 - min(X_rec_3_us10(:))) / (max(X_rec_3_us10(:)) - min(X_rec_3_us10(:)));
% X_rec_13_us10 = (X_rec_13_us10 - min(X_rec_13_us10(:))) / (max(X_rec_13_us10(:)) - min(X_rec_13_us10(:)));
% X_rec_31_us10 = (X_rec_31_us10 - min(X_rec_31_us10(:))) / (max(X_rec_31_us10(:)) - min(X_rec_31_us10(:)));
% %X_rec_3_us5 = (X_rec_3_us5 - min(X_rec_3_us5(:))) / (max(X_rec_3_us5(:)) - min(X_rec_3_us5(:)));
% X_rec_13_us5 = (X_rec_13_us5 - min(X_rec_13_us5(:))) / (max(X_rec_13_us5(:)) - min(X_rec_13_us5(:)));
% X_rec_31_us5 = (X_rec_31_us5 - min(X_rec_31_us5(:))) / (max(X_rec_31_us5(:)) - min(X_rec_31_us5(:)));
% 
% %% %%%%%%%%%% ANALYSIS %%%%%%%%%%%%%%
% 
% %Set threshhold between 0 (black) and 1 (white)
% thresh = 0.65;
% invthresh = 0.35;
% areastokeep = 20;
% 
% %% Showing binary images
% 
% % Binary masks based on thresholds
% X_ogb = (X_og < invthresh) + (X_og > thresh);
% %X_rec_3_us10b = (X_rec_3_us10 < invthresh) + (X_rec_3_us10 > thresh);
% X_rec_13_us10b = (X_rec_13_us10 < invthresh) + (X_rec_13_us10 > thresh);
% X_rec_31_us10b = (X_rec_31_us10 < invthresh) + (X_rec_31_us10 > thresh);
% %X_rec_3_us5b = (X_rec_3_us5 < invthresh) + (X_rec_3_us5 > thresh);
% X_rec_13_us5b = (X_rec_13_us5 < invthresh) + (X_rec_13_us5 > thresh);
% X_rec_31_us5b = (X_rec_31_us5 < invthresh) + (X_rec_31_us5 > thresh);
% 
% 
% hfig = figure;
% colormap gray;
% tiledlayout(2, 2, Padding="compact", TileSpacing="compact");
% 
% nexttile; imagesc(X_og); colorbar;
% 
% nexttile; imagesc(X_rec_13_us5); colorbar;
% 
% nexttile; imagesc(1-X_ogb); colorbar;
% 
% nexttile; imagesc(1-X_rec_13_us5b); colorbar;
% 
% %% Perform connected component analysis
% CC_ogb = bwconncomp(X_ogb);
% CC_rec_3_us10b = bwconncomp(X_rec_3_us10b);
% CC_rec_13_us10b = bwconncomp(X_rec_13_us10b);
% CC_rec_31_us10b = bwconncomp(X_rec_31_us10b);
% CC_rec_3_us5b = bwconncomp(X_rec_3_us5b);
% CC_rec_13_us5b = bwconncomp(X_rec_13_us5b);
% CC_rec_31_us5b = bwconncomp(X_rec_31_us5b);
% 
% % Calculate areas of connected components
% areas_ogb = cell2mat(struct2cell(regionprops(CC_ogb, "Area")));
% areas_rec_3_us10b = cell2mat(struct2cell(regionprops(CC_rec_3_us10b, "Area")));
% areas_rec_13_us10b = cell2mat(struct2cell(regionprops(CC_rec_13_us10b, "Area")));
% areas_rec_31_us10b = cell2mat(struct2cell(regionprops(CC_rec_31_us10b, "Area")));
% areas_rec_3_us5b = cell2mat(struct2cell(regionprops(CC_rec_3_us5b, "Area")));
% areas_rec_13_us5b = cell2mat(struct2cell(regionprops(CC_rec_13_us5b, "Area")));
% areas_rec_31_us5b = cell2mat(struct2cell(regionprops(CC_rec_31_us5b, "Area")));
% 
% % Sort areas in descending order
% [Asz_og, order_og] = sort(areas_ogb, 'descend');
% [Asz_rec_3_us10, order_rec_3_us10] = sort(areas_rec_3_us10b, 'descend');
% [Asz_rec_13_us10, order_rec_13_us10] = sort(areas_rec_13_us10b, 'descend');
% [Asz_rec_31_us10, order_rec_31_us10] = sort(areas_rec_31_us10b, 'descend');
% [Asz_rec_3_us5, order_rec_3_us5] = sort(areas_rec_3_us5b, 'descend');
% [Asz_rec_13_us5, order_rec_13_us5] = sort(areas_rec_13_us5b, 'descend');
% [Asz_rec_31_us5, order_rec_31_us5] = sort(areas_rec_31_us5b, 'descend');
% 
% % Plot sorted area sizes for all datasets
% figure;
% loglog(1:length(Asz_og), Asz_og, ...
%        1:length(Asz_rec_13_us10), Asz_rec_13_us10, ...
%        1:length(Asz_rec_31_us10), Asz_rec_31_us10, ...
%        1:length(Asz_rec_13_us5), Asz_rec_13_us5, ...
%        1:length(Asz_rec_31_us5), Asz_rec_31_us5, 'LineWidth', 1.5);
% legend('Original', '13 sensors (US10)', '31 sensors (US10)', ...
%        '13 sensors (US5)', '31 sensors (US5)', 'Location', 'best');
% ylim([5 max([Asz_og(1), Asz_rec_13_us10(1), Asz_rec_31_us10(1), ...
%              Asz_rec_13_us5(1), Asz_rec_31_us5(1)])]);
% title('Area sizes, darkest areas');
% ylabel('Area (pixels)');
% xlabel('Area index');
% grid on;
% 
% % Select and retain only the largest areas
% bigidx_og = order_og(1:areastokeep);
% bigidx_rec_13_us10 = order_rec_13_us10(1:areastokeep);
% bigidx_rec_31_us10 = order_rec_31_us10(1:areastokeep);
% bigidx_rec_13_us5 = order_rec_13_us5(1:areastokeep);
% bigidx_rec_31_us5 = order_rec_31_us5(1:areastokeep);
% 
% % Create binary masks for the largest areas
% X_og_tBig = cc2bw(CC_ogb, ObjectsToKeep=bigidx_og);
% X_rec_13_us10_tBig = cc2bw(CC_rec_13_us10b, ObjectsToKeep=bigidx_rec_13_us10);
% X_rec_31_us10_tBig = cc2bw(CC_rec_31_us10b, ObjectsToKeep=bigidx_rec_31_us10);
% X_rec_13_us5_tBig = cc2bw(CC_rec_13_us5b, ObjectsToKeep=bigidx_rec_13_us5);
% X_rec_31_us5_tBig = cc2bw(CC_rec_31_us5b, ObjectsToKeep=bigidx_rec_31_us5);
% 
% %% Display large figure
% 
% hfig = figure;
% %set(f,"Position",[ 183         200        900         800])
% colormap gray
% 
% tiledlayout(3,3,'Padding','compact','TileSpacing', 'compact');
% 
% nexttile, imshow(X_og), axis off,title('Original');
% nexttile, imshow(X_rec_13_us10), axis off, title('11 sensors');
% nexttile, imshow(X_rec_31_us10), axis off, title('31 sensors');
% nexttile, imshow(X_ogb), axis off;
% nexttile, imshow(X_rec_13_us10b), axis off;%, title(sprintf('Reconstruction, 11 sensors. Threshold: %d',invthresh));
% nexttile, imshow(X_rec_31_us10b), axis off;%, title(sprintf('Reconstruction, 31 sensors. Threshold: %d',invthresh));
% nexttile, imshow(X_og_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));
% nexttile, imshow(X_rec_13_us10_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));
% nexttile, imshow(X_rec_31_us10_tBig), axis off;%, title(sprintf('Threshold: %d, Only %d largest dark areas',invthresh,areastokeep));
% 
% ax = findall(hfig, 'Type', 'axes');
% set(ax, 'Box', 'on', 'XTick', [], 'YTick', []);
% 
% % Set additional properties
% fname = 'output_Nidelva/comparing_areas_3x3';
% picturewidth = 20; % Set figure width in centimeters
% hw_ratio = 1; % Height-width ratio
% 
% set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
% set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% 
% % Configure printing options
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
% 
% % Export the figure
% %print(hfig, fname, '-dpdf');%, '-vector', '-fillpage');
% %print(hfig, fname, '-dpng', '-vector');
% 
% %% Try to calculate the dark and light area coverage for every timestep and compare reconstructions
% 
% % Preallocate arrays to store the number of pixels below invthresh for each timestep
% num_pixels_ogb = zeros(1050, 1);
% num_pixels_rec_13b_us10 = zeros(1050, 1);
% num_pixels_rec_31b_us10 = zeros(1050, 1);
% num_pixels_rec_13b_us5 = zeros(1050, 1);
% num_pixels_rec_31b_us5 = zeros(1050, 1);
% 
% % Loop through all timesteps
% for timestep = 1:1050
%     disp(num2str(timestep));
%     X_og = reshape(data_og(:, timestep), nx, ny);
%     X_rec_13_us10 = reshape(data_rec_13_us10(:, timestep), nx, ny);
%     X_rec_31_us10 = reshape(data_rec_31_us10(:, timestep), nx, ny);
%     X_rec_13_us5 = reshape(data_rec_13_us5(:, timestep), nx, ny);
%     X_rec_31_us5 = reshape(data_rec_31_us5(:, timestep), nx, ny);
% 
%     % Normalize each dataset to the range [0, 1]
%     X_og = (X_og - min(X_og(:))) / (max(X_og(:)) - min(X_og(:)));
%     X_rec_13_us10 = (X_rec_13_us10 - min(X_rec_13_us10(:))) / (max(X_rec_13_us10(:)) - min(X_rec_13_us10(:)));
%     X_rec_31_us10 = (X_rec_31_us10 - min(X_rec_31_us10(:))) / (max(X_rec_31_us10(:)) - min(X_rec_31_us10(:)));
%     X_rec_13_us5 = (X_rec_13_us5 - min(X_rec_13_us5(:))) / (max(X_rec_13_us5(:)) - min(X_rec_13_us5(:)));
%     X_rec_31_us5 = (X_rec_31_us5 - min(X_rec_31_us5(:))) / (max(X_rec_31_us5(:)) - min(X_rec_31_us5(:)));
% 
%     % Binary masks based on thresholds
%     X_ogb = (X_og < invthresh) + (X_og > thresh);
%     X_rec_13_us10b = (X_rec_13_us10 < invthresh) + (X_rec_13_us10 > thresh);
%     X_rec_31_us10b = (X_rec_31_us10 < invthresh) + (X_rec_31_us10 > thresh);
%     X_rec_13_us5b = (X_rec_13_us5 < invthresh) + (X_rec_13_us5 > thresh);
%     X_rec_31_us5b = (X_rec_31_us5 < invthresh) + (X_rec_31_us5 > thresh);
% 
%     % Count the number of pixels below invthresh
%     num_pixels_ogb(timestep) = sum(X_ogb(:));
%     num_pixels_rec_13b_us10(timestep) = sum(X_rec_13_us10b(:));
%     num_pixels_rec_31b_us10(timestep) = sum(X_rec_31_us10b(:));
%     num_pixels_rec_13b_us5(timestep) = sum(X_rec_13_us5b(:));
%     num_pixels_rec_31b_us5(timestep) = sum(X_rec_31_us5b(:));
% end
% 
% %% Plot the results
% figure;
% plot(1:1050, num_pixels_rec_13b_us10, 'DisplayName', 'Reconstruction 11', 'LineWidth', 1.5); hold on;
% plot(1:1050, num_pixels_rec_31b_us10, 'DisplayName', 'Reconstruction 31', 'LineWidth', 1.5);
% plot(1:1050, num_pixels_ogb, 'DisplayName', 'Original', 'LineWidth', 1.5);
% legend('Location', 'best');
% xlabel('Timestep');
% ylabel('Number of Pixels Below invthresh');
% title('Pixels Below Threshold Over Time');
% grid on;
% hold off;
% 
% %% Define the window size for the moving average
% window_size = 20;
% 
% % Apply moving average to smooth the data
% smoothed_pixels_ogb = movmean(num_pixels_ogb, window_size);
% smoothed_pixels_rec_13b_us10 = movmean(num_pixels_rec_13b_us10, window_size);
% smoothed_pixels_rec_31b_us10 = movmean(num_pixels_rec_31b_us10, window_size);
% 
% % Plot the smoothed results
% hfig = figure;
% %plot(1:1050, smoothed_pixels_ogb, 'DisplayName', 'Original', 'LineWidth', 1.5); hold on;
% %semilogy(1:1050, smoothed_pixels_rec_3b, 'DisplayName', 'Reconstruction 3', 'LineWidth', 1.5); hold on;
% semilogy(1:1050, smoothed_pixels_rec_13b_us10, 'DisplayName', '11 sensors', 'LineWidth', 1.5); hold on;
% semilogy(1:1050, smoothed_pixels_rec_31b_us10, 'DisplayName', '31 sensors', 'LineWidth', 1.5);
% semilogy(1:1050, smoothed_pixels_ogb, 'k', 'DisplayName', 'Original', 'LineWidth', 1.5);
% xlabel('Timestep');
% ylabel('Area (pixels)');
% xlim([0,1100])
% %set(gca, 'XTick', [], 'YTick', [])
% grid on;
% 
% % Set additional properties
% fname = 'output_Nidelva/darklight_pixel_coverage_US10';
% picturewidth = 20; % Set figure width in centimeters
% hw_ratio = 0.40; % Height-width ratio
% 
% set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
% legend('Location', 'southeast', 'FontSize', 18);
% set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
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
% %print(hfig, fname, '-dpdf');%, '-vector', '-fillpage');
% %print(hfig, fname, '-dpng', '-vector');