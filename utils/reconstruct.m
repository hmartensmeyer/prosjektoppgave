function [x_hat, r, sim] = reconstruct(x, Psi, s, mean, rescale, mean_sub)
%[x_hat, r] = RECONSTRUCT(x, Theta, s, flow_params, rescale)
% Calculate normalized residual of the sparse representation of x:
%  x \approx Theta*s
% Note: Uses energy rescaling unless rescale is false

disp('HEIEIEIE')
x_hat = Psi*s;  % Sparse reconstruction

if rescale
    x_hat = x_hat*flow.avg_energy / std(x_hat); % Energy rescaling
    disp('________________________thrjiptiojh_____________________________')
end

if mean_sub
    r = norm(x-x_hat)/norm(x + mean); % Normalized L2 residual error
else
    r = norm(x-x_hat)/norm(x); % Normalized L2 residual error
end

sim = ssim(x_hat, x);

end