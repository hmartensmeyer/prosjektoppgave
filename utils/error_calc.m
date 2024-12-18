function [res, sim] = error_calc(x, x_hat, flow, mean_sub)
    % CALCULATE_ERRORS Calculate normalized L2 error and SSIM between matrices
    % Inputs:
    %   - x: Original matrix
    %   - x_hat: Reconstructed matrix
    %   - flow: Structure with mean flow data for mean subtraction
    %   - mean_sub: Boolean flag to determine if mean subtraction is needed
    %
    % Outputs:
    %   - r: Normalized L2 residual error
    %   - sim: Structural Similarity Index Measure (SSIM)

    % Calculate normalized L2 residual error
    if mean_sub
        res = norm(x - x_hat) / norm(x + flow.mean_flow); % Mean-subtracted error
    else
        res = norm(x - x_hat) / norm(x); % Standard normalized L2 error
    end

    % Calculate Structural Similarity Index Measure (SSIM)
    sim = ssim(x_hat, x);
end
