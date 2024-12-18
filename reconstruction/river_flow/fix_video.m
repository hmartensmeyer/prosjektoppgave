run("load_data.m");

%%

figure('Position', [100, 100, 500, 400]); % Adjust the dimensions as needed

for t = 137:1000
    imagesc(reshape(data_normalized(:, t), nx, ny));
    %clim([50 200])
    colormap gray;
    colorbar;
    title(num2str(t))
    pause(0.02);
end

%% SVD of data_reshaped and then throw away first mode

% Perform SVD on the reshaped data
[U, Sigma, V] = svd(data_normalized - mean(data_normalized,2), 'econ');

%% Choose the number of modes to plot and reconstruct
num_modes = 5; % Adjust this to plot and reconstruct different numbers of modes

% Plot the first few singular modes
figure('Position', [100, 100, 1200, 400]); % Wide and short figure for better visualization
for k = 1:num_modes
    subplot(1, num_modes, k);
    imagesc(reshape(U(:, k), 540, 540)); % Reshape to original spatial dimensions
    colormap gray;
    colorbar;
    title(['Mode ', num2str(k)]);
end

%%
% Extract singular values from the diagonal of Sigma
singular_values = diag(Sigma);

% Plot the singular values
figure;
loglog(singular_values / sum(singular_values), 'o-', 'LineWidth', 1.5);
xlabel('Index');
ylabel('Singular Value');
title('Singular Values of data\_reshaped');
grid on;

% Plot cumsum
figure;
semilogx(cumsum(singular_values) / sum(singular_values), 'o');
xlabel('r');
ylabel('Cumulative sum');
grid on;


%%
starter = 1;
ender = 700;

% Reconstruct the data using the first few modes
data_reconstructed = U(:, starter:ender) * Sigma(starter:ender, starter:ender) * V(:, starter:ender)';

% Verify reconstruction quality by plotting the first frame
figure;
imagesc(reshape(data_reconstructed(:, 1), nx, ny));
colormap gray;
colorbar;
title(['Reconstructed Frame 1 using ', num2str(num_modes), ' modes']);

%% Normalization of SVD-recon

% Min-Max Normalization per column
min_vals = min(data_reconstructed, [], 1);
max_vals = max(data_reconstructed, [], 1);
epsilon = 1e-8; % To avoid division by zero
data_reconstructed = (data_reconstructed - min_vals) ./ (max_vals - min_vals + epsilon);
% %%
% 
% figure('Position', [100, 100, 500, 400]); % Adjust the dimensions as needed
% 
% for t = 1:1000
%     imagesc(reshape(data_reconstructed(:, t), nx, ny));
%     %clim([0, 0.8])
%     colormap gray;
%     colorbar;
%     pause(0.02);
% end
% 
% %% Trying to cap to get constrast
% 
% % Set the lower and upper caps
% lower_cap = 0.3;
% upper_cap = 0.85;
% 
% % Apply the lower and upper caps
% video_capped = data_reconstructed;
% video_capped(data_reconstructed < lower_cap) = lower_cap;
% video_capped(data_reconstructed > upper_cap) = upper_cap;
% 
% % Display or process `video_capped` as needed
% 
% %% Normalization of SVD-recon
% 
% % Min-Max Normalization per column
% min_vals = min(video_capped, [], 1);
% max_vals = max(video_capped, [], 1);
% epsilon = 1e-8; % To avoid division by zero
% video_capped = (video_capped - min_vals) ./ (max_vals - min_vals + epsilon);
% %%
% 
% figure('Position', [100, 100, 500, 400]); % Adjust the dimensions as needed
% 
% for t = 1:1000
%     imagesc(reshape(video_capped(:, t), nx, ny));
%     colormap gray;
%     colorbar;
%     pause(0.02);
% end
