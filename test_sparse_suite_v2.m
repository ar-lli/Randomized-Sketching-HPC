1;

%% -------------------- Utilities --------------------
function y = select_indices(x, I)
    y = x(I);
endfunction

function S = count_sketch(m, n)
    target_rows = randi(n, m, 1);
    signs = 2 * (rand(m, 1) > 0.5) - 1;
    S = sparse(1:m, target_rows, signs, m, n);
endfunction

function S = srht_sketch(n, k)
    m = 2^nextpow2(n);
    d = sign(randn(m,1));
    idx = randperm(m,k);
    scale = sqrt(m/k);
    S = @(x) scale * fwht_apply(d, idx, x, n, m);
endfunction

function y = fwht_apply(d, idx, x, n, m)
    xp = zeros(m,1);
    xp(1:n) = x;
    xp = d .* xp;
    xp = fwht(xp);
    y = xp(idx);
endfunction

function S = sdct_sketch(n, k)
    d = sign(randn(n,1));
    idx = randperm(n,k);
    scale = sqrt(n/k);
    S = @(x) scale * dct(d .* x)(idx);
endfunction

%% -------------------- Matrix List --------------------
matrices_file = {'poisson3Da', 'poisson3Db'};
maxit = 500;

%% -------------------- Plot Layout --------------------
rows = 1; cols = 2;
hmargin = 0.08; vmargin = 0.10; hgap = 0.06; vgap = 0.12;
plot_box_width  = (1-2*hmargin-(cols-1)*hgap)/cols;
plot_box_heigth = (1-2*vmargin-(rows-1)*vgap)/rows;

figure('name','Residual / Iterations for Sparse Matrices'); hold on;

for j = 1:length(matrices_file)
    A_struct = load(strcat('sparse_matrices/', char(matrices_file(j)), '.mat'));
    A = A_struct.Problem.A;
    n = size(A,1);
    fprintf("Matrix %s: size n = %d\n", char(matrices_file(j)), n);

    x_true = ones(n,1);
    b = A * x_true;

    %% --- Grid subplot position ---
    row_idx = ceil(j / cols);
    col_idx = mod(j-1, cols) + 1;
    x_pos = hmargin + (col_idx-1)*(plot_box_width+hgap);
    y_pos = 1 - vmargin - row_idx*plot_box_heigth - (row_idx-1)*vgap;
    ax = subplot('Position', [x_pos, y_pos, plot_box_width, plot_box_heigth]);

    %% --- GMRES baseline ---
    tt = tic;
    [~, res0] = my_gmres(A,b,1e-6,maxit);
    fprintf("GMRES: %.4f s\n", toc(tt));

    %% --- Sketch sizes ---
    k_gauss = 2*maxit + 1;
    k_large = 6*maxit + 1;

    %% --- Gaussian ---
    tt = tic;
    S1 = randn(n, k_gauss) / sqrt(n);
    [~, res1] = my_gmres_sketch(A,b,1e-6,maxit,S1);
    fprintf("Gaussian: %.4f s\n", toc(tt));

    %% --- SRHT (Fast Hadamard) ---
    tt = tic;
    S2 = srht_sketch(n, k_gauss);
    [~, res2] = my_gmres_sketch(A,b,1e-6,maxit,S2);
    fprintf("SRHT: %.4f s\n", toc(tt));

    %% --- DCT ---
    tt = tic;
    S3 = sdct_sketch(n, k_gauss);
    [~, res3] = my_gmres_sketch(A,b,1e-6,maxit,S3);
    fprintf("DCT: %.4f s\n", toc(tt));

    %% --- Rademacher ---
    tt = tic;
    S4 = (rand(n,k_gauss)>0.5)*2 - 1;
    [~, res4] = my_gmres_sketch(A,b,1e-6,maxit,S4);
    fprintf("Rademacher: %.4f s\n", toc(tt));

    %% --- SRFT (Fourier) ---
    tt = tic;
    I5 = randsample(n, k_large);
    S5 = @(v) select_indices(fft(v), I5);
    [~, res5] = my_gmres_sketch(A,b,1e-6,maxit,S5);
    fprintf("SRFT: %.4f s\n", toc(tt));

    %% --- CountSketch ---
    tt = tic;
    S6 = count_sketch(n, k_large);
    [~, res6] = my_gmres_sketch(A,b,1e-6,maxit,S6);
    fprintf("CountSketch: %.4f s\n", toc(tt));

    %% --- Plotting ---
    hold on;
    semilogy(res0,'k--','LineWidth',2);
    semilogy(res1); semilogy(res2); semilogy(res3);
    semilogy(res4); semilogy(res5); semilogy(res6);

    grid on;
    title(sprintf('%s, n=%d', char(matrices_file(j)), n));
    if col_idx==1, ylabel('Residual'); end
    if row_idx==rows, xlabel('Iterations'); end

    if j==1
        legend('GMRES','Gaussian','SRHT','DCT','Rademacher','SRFT','CountSketch',...
               'Location','southwest','FontSize',8);
    end
    set(ax,'LooseInset',get(ax,'TightInset'));
endfor
hold off;

print(fig1, './plots/tests_sparse_suite.png', '-dpng');