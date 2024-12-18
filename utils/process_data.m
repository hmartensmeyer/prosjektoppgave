% Function to process data and compute SVD and energy
function [singular_values, normalized_singular_values, cumulative_energy, U, S, V] = process_data(data, mean_dim)
    % Compute mean along specified dimension
    mean_data = mean(data, mean_dim);

    % Subtract mean from data
    data_centered = data - mean_data;

    % Perform SVD
    [U, S, V] = svd(data_centered, 'econ');

    % Calculate singular values, normalized singular values, cumulative energy
    singular_values = diag(S);
    normalized_singular_values = singular_values / sum((singular_values));
    cumulative_energy = cumsum(singular_values) / sum(singular_values);

    U = U;
end