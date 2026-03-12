% MATLAB script: test_sparse_suite_v2.m

%% -------------------- Matrix List --------------------
matrices_file = {'poisson3Da', 'poisson3Db'};
maxit = 500;

%% -------------------- Plot Layout --------------------
rows = 1; cols = 2;
hmargin = 0.08; vmargin = 0.10; hgap = 0.06; vgap = 0.12;
plot_box_width  = (1-2*hmargin-(cols-1)*hgap)/cols;
plot_box_heigth = (1-2*vmargin-(rows-1)*vgap)/rows;

fig1 = figure('Name','Residual / Iterations for Sparse Matrices');

for j = 1:length(matrices_file)
    % MATLAB load creates a struct
    A_struct = load(fullfile('sparse_matrices', [char(matrices_file(j)), '.mat']));
    A = A_struct.Problem.A;
    n = size(A,1);
    fprintf('Matrix %s: size n = %d\n', char(matrices_file(j)), n);

    x_true = ones(n,1);
    b = A * x_true;

    %% --- Grid subplot position ---
    row_idx = ceil(j / cols);
    col_idx = mod(j-1, cols) + 1;
    x_pos = hmargin + (col_idx-1)*(plot_box_width+hgap);
    y_pos = 1 - vmargin - row_idx*plot_box_heigth - (row_idx-1)*vgap;
    ax = subplot('Position', [x_pos, y_pos, plot_box_width, plot_box_heigth]);

    %% --- Solver Calls ---
    % GMRES baseline
    tt = tic; [~, r0] = my_gmres(A,b,1e-6,maxit); fprintf('GMRES: %.4f s\n', toc(tt));
    
    k_gauss = 2*maxit + 1;
    k_large = 6*maxit + 1;

    % Sketched variants
    tt = tic; S1 = randn(n, k_gauss) / sqrt(n); [~, r1] = my_gmres_sketch(A,b,1e-6,maxit,S1); fprintf('Gaussian: %.4f s\n', toc(tt));
    tt = tic; S2 = srht_sketch(n, k_gauss); [~, r2] = my_gmres_sketch(A,b,1e-6,maxit,S2); fprintf('SRHT: %.4f s\n', toc(tt));
    tt = tic; S3 = sdct_sketch(n, k_gauss); [~, r3] = my_gmres_sketch(A,b,1e-6,maxit,S3); fprintf('DCT: %.4f s\n', toc(tt));
    tt = tic; S4 = (rand(n,k_gauss)>0.5)*2 - 1; [~, r4] = my_gmres_sketch(A,b,1e-6,maxit,S4); fprintf('Rademacher: %.4f s\n', toc(tt));
    tt = tic; I5 = randsample(n, k_large); S5 = @(v) select_indices(fft(v), I5); [~, r5] = my_gmres_sketch(A,b,1e-6,maxit,S5); fprintf('SRFT: %.4f s\n', toc(tt));
    tt = tic; S6 = count_sketch(k_large, n); [~, r6] = my_gmres_sketch(A,b,1e-6,maxit,S6); fprintf('CountSketch: %.4f s\n', toc(tt));

    %% --- Plotting ---
    hold on;
    semilogy(r0,'k--','LineWidth',2);
    semilogy(r1); semilogy(r2); semilogy(r3);
    semilogy(r4); semilogy(r5); semilogy(r6);

    grid on;
    title(sprintf('%s, n=%d', char(matrices_file(j)), n));
    if col_idx==1, ylabel('Residual'); end
    if row_idx==rows, xlabel('Iterations'); end

    if j==1
        legend('GMRES','Gaussian','SRHT','DCT','Rademacher','SRFT','CountSketch',...
               'Location','southwest','FontSize',8);
    end
end

print(fig1, 'tests_sparse_suite.png', '-dpng');

%% -------------------- Utilities (Moved to End) --------------------
function y = select_indices(x, I), y = x(I); end

function S = count_sketch(m, n)
    target_rows = randi(n, m, 1); signs = 2 * (rand(m, 1) > 0.5) - 1;
    S = sparse(1:m, target_rows, signs, m, n);
end

function S = srht_sketch(n, k)
    m = 2^nextpow2(n); d = sign(randn(m,1)); idx = randperm(m,k); scale = sqrt(m/k);
    S = @(x) scale * fwht_apply(d, idx, x, n, m);
end

function y = fwht_apply(d, idx, x, n, m)
    xp = zeros(m,1); xp(1:n) = x; xp = d .* xp; xp = fwht(xp); y = xp(idx);
end

function S = sdct_sketch(n, k)
    d = sign(randn(n,1)); idx = randperm(n,k); scale = sqrt(n/k);
    S = @(x) scale * apply_dct_idx(d, idx, x);
end
function res = apply_dct_idx(d, idx, x)
    tmp = dct(d .* x); res = tmp(idx);
end