% Plotting Script: Load saved SVD results and generate plots

% Load the SVD results
load('output/svd_results.mat');

%%

disp(normalized_singular_values_all.VS(1))

% %%
% % Testing new plotting
% PS = PLOT_STANDARDS();
% 
% % Plotting
% hfig = figure;
% 
% superfig.t1 = tiledlayout(hfig, 2, 1);
% 
% % Subplot for normalized singular values
% %subplot(2,1,1);
% nexttile;
% loglog(normalized_singular_values_all.VS, 'o', 'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Red4); hold on;
% loglog(normalized_singular_values_all.VS_higher_Re, 'o', 'DisplayName', 'Cylinder, Re = 7000', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Blue4);
% %loglog(normalized_singular_values_all.VS_higher_Re_noise, 'o-', 'DisplayName', 'Cylinder flow, higher Re, with noise', 'MarkerSize', 3);
% %loglog(normalized_singular_values_all.SST, 'o-', 'DisplayName', 'SST', 'MarkerSize', 3);
% loglog(normalized_singular_values_all.DNS, 'o', 'DisplayName', 'Surface elevation, DNS', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Green4);
% %loglog(normalized_singular_values_all.DNS_U / normalized_singular_values_all.DNS_U(1), 'o-', 'DisplayName', 'Surface velocity, DNS', 'MarkerSize', 3);
% loglog(normalized_singular_values_all.nidelva, 'o', 'DisplayName', 'Nidelva', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Orange4);
% 
% %xlabel('$r$', 'FontWeight','bold');%,'FontSize',FS, 'interpreter', 'latex')
% ylabel('$\sigma_r$', 'FontWeight','bold')
% set(gca, 'XTickLabel', []);  % Remove x-axis tick labels
% ylim([1e-9 1]);
% legend('Location', 'southwest', 'FontSize', 18, 'FontWeight','bold');
% grid on;
% 
% % Subplot for cumulative energy
% %subplot(2,1,2);
% nexttile;
% semilogx(cumulative_energy_all.VS, 'o', 'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Red4); hold on;
% semilogx(cumulative_energy_all.VS_higher_Re, 'o', 'DisplayName', 'Cylinder flow, Re = 7000', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Blue4);
% %semilogx(cumulative_energy_all.VS_higher_Re_noise, 'o-', 'DisplayName', 'Cylinder flow, higher Re, with noise', 'MarkerSize', 3);
% %semilogx(cumulative_energy_all.SST, 'o-', 'DisplayName', 'SST', 'MarkerSize', 3);
% semilogx(cumulative_energy_all.DNS, 'o', 'DisplayName', 'Surface elevation, DNS', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Green4);
% %semilogx(cumulative_energy_all.DNS_U, 'o-', 'DisplayName', 'Surface velocity, DNS', 'MarkerSize', 3);
% %legend('Location', 'southeast', 'FontSize', 18, 'FontWeight','bold');
% semilogx(cumulative_energy_all.nidelva, 'o', 'DisplayName', 'Nidelva', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Orange4);
% 
% xlabel('$r$');%, 'FontSize', 14);
% ylabel('Cumulative sum');%, 'FontSize', 14);
% ylim([0, 1.05]);
% %legend;
% grid on;
% 
% superfig.t1.Padding = 'tight';
% set(gca, 'yticklabel', []); % Removes yticklabels for the current axis. Use as in Tile 4 to remove for each individual tile.
% superfig.t1.TileSpacing = 'none';
% 
% fname = 'output/combined_all_datasets';
% 
% picturewidth = 20; % set this parameter and keep it forever
% hw_ratio = 1; % feel free to play with this ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',21) % adjust fontsize to your document
% %legend('Location', 'southeast', 'FontSize', 18, 'FontWeight','bold');
% set(findall(hfig,'-property','Box'),'Box','off') % optional
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
% box on;
% 
% %print(hfig,fname,'-dpdf','-painters','-fillpage')
% %print(hfig,fname,'-dpng','-vector')

%% Refined plot

% Testing new plotting
PS = PLOT_STANDARDS();

% Plotting
hfig = figure;

superfig.fig = gcf;

% Set up a 2x1 tiled layout with tight padding and no spacing
superfig.t1 = tiledlayout(hfig, 2, 1, 'Padding', 'tight', 'TileSpacing', 'none');

% First tile for normalized singular values
superfig.n(1) = nexttile;
loglog(normalized_singular_values_all.VS, 'o', 'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Red4); hold on;
loglog(normalized_singular_values_all.VS_higher_Re, 'o', 'DisplayName', 'Cylinder, Re = 7000', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Orange4);
loglog(normalized_singular_values_all.DNS, 'o', 'DisplayName', 'Surface elevation, DNS', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Green4);
loglog(normalized_singular_values_all.nidelva, 'o', 'DisplayName', 'Nidelva', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Blue4);

ylabel('$\sigma_r$', 'FontWeight', 'bold');
set(gca, 'XTickLabel', []);  % Remove x-axis tick labels
ylim([1e-7 1]);
%legend('Location', 'southwest', 'FontSize', 10, 'FontWeight', 'bold');
grid on;

% Second tile for cumulative energy
superfig.n(2) = nexttile;
semilogx(cumulative_energy_all.VS, 'o', 'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Red4); hold on;
semilogx(cumulative_energy_all.VS_higher_Re, 'o', 'DisplayName', 'Cylinder, Re = 7000', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Orange4);
semilogx(cumulative_energy_all.DNS, 'o', 'DisplayName', 'Surface elevation, DNS', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Green4);
semilogx(cumulative_energy_all.nidelva, 'o', 'DisplayName', 'Nidelva', 'MarkerSize', 5, 'MarkerEdgeColor', PS.Blue4);

xlabel('$r$', 'FontWeight', 'bold');
ylabel('Cumulative sum', 'FontWeight', 'bold');
ylim([0, 1.05]);
grid on;

fname = 'output/MERGED_FIG';

% Set figure properties
set(findall(hfig, '-property', 'FontSize'), 'FontSize', 21); % Adjust fontsize
set(findall(hfig, '-property', 'Box'), 'Box', 'on'); % Remove box if needed
set(findall(hfig, '-property', 'Interpreter'), 'Interpreter', 'latex');
set(findall(hfig, '-property', 'TickLabelInterpreter'), 'TickLabelInterpreter', 'latex');
legend('Location', 'southeast', 'FontSize', 14, 'FontWeight', 'bold', 'Box','off');

% Configure figure size and save options
picturewidth = 20; % in centimeters
hw_ratio = 0.7; % height-width ratio
set(hfig, 'Units', 'centimeters', 'Position', [3 3 picturewidth hw_ratio * picturewidth]);
pos = get(hfig, 'Position');
set(hfig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'centimeters', 'PaperSize', [pos(3), pos(4)]);

% Uncomment the appropriate line below to save
print(hfig, fname, '-dpdf', '-vector', '-fillpage');
%print(hfig, fname, '-dpng', '-r300'); % Adjust resolution if needed

%% Display r_90 values
disp('Number of modes to reach 90% cumulative energy (r_90):');
disp(r_90_all);

%% In two different plots

% Define common properties for the publication-quality plot
marker_size = 4;
line_width = 1.5;
font_size = 14;

% Plot 1: Normalized Singular Values
hfig = figure;% (Position = ([300,300,900,400]));
loglog(normalized_singular_values_all.VS, 'o', ...
    'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', marker_size, 'LineWidth', line_width); hold on;
loglog(normalized_singular_values_all.VS_higher_Re, 'o', ...
    'DisplayName', 'Cylinder, Re = 7000', 'MarkerSize', marker_size, 'LineWidth', line_width);
loglog(normalized_singular_values_all.DNS, 'o', ...
    'DisplayName', 'Surface elevation, DNS', 'MarkerSize', marker_size, 'LineWidth', line_width);
loglog(normalized_singular_values_all.nidelva, 'o', ...
    'DisplayName', 'Nidelva', 'MarkerSize', marker_size, 'LineWidth', line_width);

%xlabel('Mode Number', 'FontSize', font_size);%, 'FontWeight', 'bold');
%ylabel('Normalized Singular Values', 'FontSize', font_size);%, 'FontWeight', 'bold');
xlabel('$r$', 'FontWeight','bold');%,'FontSize',FS, 'interpreter', 'latex')
%ylabel('$\sigma_r/\Sigma\sigma_r$');%,'FontSize',FS, 'interpreter', 'latex')
ylabel('$\sigma_r$', 'FontWeight','bold')

ylim([1e-10 1]);
%xlim([-1 6000])
legend('Location', 'southwest', 'FontSize', 18);
%set(gca, 'FontSize', font_size, 'LineWidth', line_width);
% Define custom y-ticks for the grid on the log scale
yticks = logspace(-10, 0, 11); % Adjust the range and number of ticks as needed
set(gca, 'YTick', yticks(1:2:end)); % Apply the custom y-ticks

grid on;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'off', 'YGrid', 'on'); % Enforce minor grid settings
box on;
fname = 'output/SVD_all_datasets_nonnormal';

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
print(hfig,fname,'-dpdf','-vector');%, '-bestfit');%,'-fillpage')
%print(hfig,fname,'-dpng','-vector')

%%
% Plot 2: Cumulative Energy
hfig = figure(2);
semilogx((cumulative_energy_all.VS), 'o', ...
    'DisplayName', 'Cylinder, Re = 100', 'MarkerSize', marker_size, 'LineWidth', line_width); hold on;
semilogx(cumulative_energy_all.VS_higher_Re, 'o', ...
    'DisplayName', 'Cylinder, Re = 7000', 'MarkerSize', marker_size, 'LineWidth', line_width);
semilogx(cumulative_energy_all.DNS, 'o', ...
    'DisplayName', 'Surface elevation, DNS', 'MarkerSize', marker_size, 'LineWidth', line_width);
semilogx(cumulative_energy_all.nidelva, 'o', ...
    'DisplayName', 'Nidelva', 'MarkerSize', marker_size, 'LineWidth', line_width);

xlabel('$r$', 'FontSize', font_size);%, 'FontWeight', 'bold');
ylabel('Cumulative sum', 'FontSize', font_size);%, 'FontWeight', 'bold');
%legend('Location', 'southwest', 'FontSize', font_size);
%set(gca, 'FontSize', font_size, 'LineWidth', line_width);
grid on;
%box on;
ylim([0, 1.05])
xlim([0 3e4])
set(gca, 'XScale', 'Log');
set(gca, 'XTick', [10^0, 10^1, 10^2, 10^3, 10^4]);

fname = 'output/CUMSUM_all_datasets';

picturewidth = 20; % set this parameter and keep it forever
hw_ratio = 0.55; % feel free to play with this ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20) % adjust fontsize to your document
legend('Location', 'southeast', 'FontSize', 18, 'FontWeight','bold');
set(findall(hfig,'-property','Box'),'Box','off') % optional
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex') 
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex')
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth])
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)])
box on;
print(hfig,fname,'-dpdf','-vector');%,'-fillpage')
%print(hfig,fname,'-dpng','-vector')

