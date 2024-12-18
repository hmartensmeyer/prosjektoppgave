function [PSD_avg, f] = PSD_calc(data, fs)
    % compute_average_psd_single_dataset computes and plots the average PSD of a dataset.
    %
    % Parameters:
    %   data         - Data matrix [n_points x n_samples]
    %   fs           - Sampling frequency in Hz
    %
    % Example usage:
    %   compute_average_psd_single_dataset(X_noiseless, 15, 'output/PSD_Re7000_X_noiseless', 'Original flow field');

    % Get the dimensions of the data
    [n, m] = size(data);

    % % Adjust Welch's method parameters based on data size
    % max_window_length = min(512, n);
    % max_power = floor(log2(max_window_length));       % Find the largest power of two less than or equal to max_window_length
    % window_length = 2^max_power;                      % Window length as a power of two
    % window = hann(window_length);                     % Hanning window
    % noverlap = floor(window_length / 2);              % 50% overlap
    % nfft = window_length; 

    % Define Welch's method parameters
    window_length = 1024;            % Window length
    window = hann(window_length);    % Hanning window
    noverlap = 512;                  % 50% overlap
    nfft = 2048;                      % Number of FFT points

    % Initialize variables to accumulate PSDs
    PSD_total = zeros(nfft/2 + 1, 1);

    % Loop through all spatial points
    for i = 1:n
        % Extract the time series at point i
        signal = data(i, :);

        disp([num2str(i), ' / ', num2str(n)]);

        % Compute PSD using Welch's method
        [Pxx, f] = pwelch(signal, window, noverlap, nfft, fs);

        % Accumulate the PSDs
        PSD_total = PSD_total + Pxx;
    end

    % Calculate the average PSD
    PSD_avg = PSD_total / n;
    f = f;
end
