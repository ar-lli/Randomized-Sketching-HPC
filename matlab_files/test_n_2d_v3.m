% MATLAB script: test_n_2d_v3.m
% Note: Requires Statistics and Signal Processing Toolboxes

%% -------------------- Experiment Setup --------------------
n_dimensions = [50, 70];
rows = 1; cols = 2;
maxit = 50;
eta = 0.5;

methods = {'GMRES','Gaussian','Hadamard','DCT','Rademacher','SRFT','CountSketch'};
nm = length(methods);

times = zeros(length(n_dimensions), nm);

% Assign handle to figure
fig1 = figure('Name','Residual vs Iterations');

%% -------------------- Main Loop --------------------
for j = 1:length(n_dimensions)

    n = n_dimensions(j);
    fprintf('Case n = %d\n\n', n);

    h = 1/(n+1);
    t = linspace(0,1,n+2);
    [xx,yy] = meshgrid(t,t);

    X_true = -sin(xx.*pi).*exp(yy).*(yy.^2 - yy);
    x_true = X_true(2:end-1,2:end-1);
    x_true = x_true(:);

    A_poisson = gallery('poisson',n)/h^2;
    D = spdiags(ones(n^2,1)*[-1 1],[-1,1],n^2,n^2)/(2*h);

    A = A_poisson + eta*D;
    b = A*x_true;

    k_gauss = 2*maxit + 1;
    k_large = 8*maxit + 1;

    %% -------- GMRES --------
    tt = tic();
    [~,res_cell{1}] = my_gmres(A,b,1e-6,maxit);
    times(j,1) = toc(tt);

    %% -------- Sketches --------
    sketches = {
        randn(n^2,k_gauss)/sqrt(n^2),
        srht_sketch(n^2,k_gauss),
        sdct_sketch(n^2,k_gauss),
        (rand(n^2,k_gauss)>0.5)*2-1,
        @(v) select_indices(fft(v), randsample(n^2,k_large)),
        count_sketch(n^2,k_large)
    };

    for m = 2:nm
        tt = tic();
        [~,res_cell{m}] = my_gmres_sketch(A,b,1e-6,maxit,sketches{m-1});
        times(j,m) = toc(tt);
        fprintf('%s: %.3f sec\n', methods{m}, times(j,m));
    end

    %% -------- Plot Iterations/Residuals--------
    figure(fig1);
    subplot(rows, cols, j)
    hold on

    styles = {'k--','b','r','g','m','c','y'};

    for m = 1:nm
        semilogy(res_cell{m}, styles{m}, 'LineWidth', 1.2);
    end

    grid on
    title(sprintf('n = %d^2', n))

    if j == 1
        legend(methods,'Location','southwest','FontSize',8)
    end
    if mod(j-1,cols)==0
        ylabel('Residual')
    end
    if j > (rows-1)*cols
        xlabel('Iterations')
    end
end

% Save the convergence plot
print(fig1, 'test_n_2d.png', '-dpng');

%% -------- Plot Runtime Scaling --------
fig2 = figure('Name','Runtime Scaling');
hold on
colors = lines(nm);

for m = 1:nm
    plot(n_dimensions, times(:,m), '-o', ...
        'Color', colors(m,:), ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6);
end

grid on
xlabel('Matrix dimension n')
ylabel('Execution time (seconds)')
title('Runtime scaling of GMRES variants')
legend(methods,'Location','northwest')
hold off

% Save the runtime plot
print(fig2, 'test_n_2d_runtime.png', '-dpng');

%% -------------------- Timing Summary --------------------
T = array2table(times, 'VariableNames', methods, 'RowNames', string(n_dimensions));
disp('Execution times (seconds):')
disp(T)
writetable(T, 'test_n_2d_runtime.csv');

%% -------------------- Utilities (Moved to End) --------------------
function y = select_indices(x, I), y = x(I); end

function S = sdct_sketch(n, k)
    d = sign(randn(n,1)); idx = randperm(n, k); scale = sqrt(n/k);
    S = @(x) scale * apply_dct_idx(d, idx, x);
end
function res = apply_dct_idx(d, idx, x)
    tmp = dct(d .* x); res = tmp(idx);
end

function S = srht_sketch(n, k)
    m = 2^nextpow2(n); d = sign(randn(m,1)); idx = randperm(m, k); scale = sqrt(m/k);
    S = @(x) scale * fwht_apply(d, idx, x, n, m);
end

function y = fwht_apply(d, idx, x, n, m)
    xp = zeros(m,1); xp(1:n) = x; xp = d .* xp; xp = fwht(xp); y = xp(idx);
end

function S = count_sketch(m, n)
    target_rows = randi(n, m, 1); signs = 2 * (rand(m,1) > 0.5) - 1;
    S = sparse(1:m, target_rows, signs, m, n);
end