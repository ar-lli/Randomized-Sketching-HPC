# This test compares performance of GMRES and sGMRES
# against matrices from the SuiteSparse / gallery collection

1;

# --- Function Definitions ---
function y = select_indices(x, I)
    y = x(I);
endfunction

function S = count_sketch(m, n)
    target_rows = randi(n, m, 1);
    signs = 2 * (rand(m,1) > 0.5) - 1;
    S = sparse(1:m, target_rows, signs, m, n);
endfunction


# --- Experiment Parameters ---
n_vec = [100, 120, 140, 160];
matrices = {'dorr'}; 

rows = 2;
cols = 2;

figure('name', 'Residual vs Iterations');

# --- Main experiment loops ---
for i = 1:length(matrices)
for j = 1:length(n_vec)

    n = n_vec(j);

    subplot(rows, cols, (i-1)*cols + j);
    hold on;

    # --- Matrix definition ---
    A = gallery(char(matrices(i)), n^2);
    x_true = ones(n^2,1);
    b = A * x_true;

    maxit = round(n^2 * 0.07);

    # --- 1. GMRES ---
    tt = tic();
    [~, res0] = my_gmres(A, b, 1e-6, maxit);
    fprintf('GMRES (%s, n=%d): %f sec\n', char(matrices(i)), n, toc(tt));

    # --- Sketch dimensions ---
    k_gauss = 2*maxit + 1;
    k_large = 4*maxit + 1;

    # --- 2. Gaussian sketch ---
    tt = tic();
    S1 = randn(n^2, k_gauss) / sqrt(k_gauss);
    [~, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);
    fprintf('Gaussian: %f sec\n', toc(tt));

    # --- 3. Rademacher sketch ---
    tt = tic();
    S3 = (rand(n^2, k_gauss) > 0.5)*2 - 1;
    [~, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);
    fprintf('Rademacher: %f sec\n', toc(tt));

    # --- 4. SRFT sketch ---
    tt = tic();
    I4 = randsample(n^2, k_large);
    S4 = @(v) select_indices(fft(v), I4);
    [~, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);
    fprintf('SRFT: %f sec\n', toc(tt));

    # --- 5. CountSketch ---
    tt = tic();
    S5 = count_sketch(n^2, k_large);
    [~, res5] = my_gmres_sketch(A, b, 1e-6, maxit, S5);
    fprintf('CountSketch: %f sec\n', toc(tt));

    # --- Plot ---
    semilogy(res0,'k--','LineWidth',2);
    semilogy(res1);
    semilogy(res3);
    semilogy(res4);
    semilogy(res5);

    grid on;

    title(sprintf('%s  (n=%d)', char(matrices(i)), n));

    if j == 1
        ylabel('Residual');
    endif

    if i == rows
        xlabel('Iterations');
    endif

    if i == 1 && j == 1
        legend('GMRES','Gaussian','Rademacher','SRFT','CountSketch',...
               'Location','southwest','FontSize',8);
    endif

endfor
endfor