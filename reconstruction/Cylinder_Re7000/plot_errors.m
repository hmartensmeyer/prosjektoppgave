data_rand = load('output\Re7000_US_random.mat');
data_qr = load('output\Re7000_US_QR.mat');
data3 = load('output\Re100_US_random.mat');
data4 = load("output\Re100_US_QR.mat");
data5 = load('..\DNS\output\DNS_US_random.mat');
data6 = load('..\DNS\output\DNS_US_QR.mat');
data7 = load('..\Nidelva\output\nidelva_US_QR.mat');

%%
figure;
plot(data_rand.us_values, data_rand.mean_residuals, DisplayName='Re7000, 30 random sensors', LineWidth=2);
hold on;
plot(data_qr.us_values, data_qr.mean_residuals, DisplayName='Re7000, 30 QR sensors', LineWidth=2);
plot(data3.us_values, data3.mean_residuals, DisplayName='Re100, 10 random sensors', LineWidth=2);
plot(data4.us_values, data4.mean_residuals, DisplayName='Re100, 10 QR sensors', LineWidth=2);
plot(data5.us_values, data5.mean_residuals, DisplayName='DNS, 30 random sensors', LineWidth=2);
plot(data6.us_values, data6.mean_residuals, DisplayName='DNS, 30 QR sensors', LineWidth=2);
plot(data7.us_values, data7.mean_residuals, DisplayName='Nidelva, 30 QR sensors', LineWidth=2);
hold off;

% Adding labels and title
xlabel('Undersampling rate');
ylabel('Error');

% Display the legend
legend('show');

% Add grid for better readability
grid on;

%%

% Plotting the residuals
figure;
plot(dt5_sens5, 'DisplayName', 'dt5\_sens5', 'LineWidth', 2);
hold on;
plot(dt5_sens50, 'DisplayName', 'dt5\_sens50', 'LineWidth', 2);
plot(re_100_dt5_sens50, 'DisplayName', 're=100_dt5\_sens50', 'LineWidth', 2);
plot(re_100_dt5_sens5, 'DisplayName', 're=100_dt5\_sens5', 'LineWidth', 2);
plot(dt5_sens5_QR, 'DisplayName', 'dt5\_sens5_QR', 'LineWidth', 2);
plot(dt5_sens50_QR, 'DisplayName', 'dt5\_sens50_QR', 'LineWidth', 2);
hold off;

% Adding labels and title
xlabel('Time Step');
ylabel('Residual');
title('Reconstruction Residuals Comparison');

% Display the legend
legend('show');

% Add grid for better readability
grid on;

%%

disp(mean(dt5_sens5));
disp(mean(dt5_sens50));
disp(mean(re_100_dt5_sens50));
disp(mean(re_100_dt5_sens5));
disp(mean(dt5_sens5_QR));
disp(mean(dt5_sens50_QR));

%%

data1 = load('output/re_7000_res_dt5_sens20_random.mat');
data2 = load('output/re_7000_res_dt5_sens20_QR.mat');

%%

data1 = data1.residuals;
data2 = data2.residuals;

%%

figure;
plot(data1, 'DisplayName', 'random sensors', 'LineWidth', 2);
hold on;
plot(data2, 'DisplayName', 'QR sensors', 'LineWidth', 2);

%Adding labels and title
xlabel('Time Step');
ylabel('Residual');
title('Reconstruction Residuals Comparison');

% Display the legend
legend('show');

% Add grid for better readability
grid on;

%%

disp(mean(data1));
disp(mean(data2));