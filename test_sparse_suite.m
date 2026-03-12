# This test compares performace of GMRES and sGMRES algorithm against matrices of the Sparse Suite Matrix collection

1;

# --- Function Definitions ---
function y = select_indices(x, I)
    y = x(I);
endfunction

function S = count_sketch(m, n)
# m: original (large) dimension, n: reduced (target) dimension
    target_rows = randi(n, m, 1);
    signs = 2 * (rand(m, 1) > 0.5) - 1;
# S is m x n such that S' * x performs the sketching
    S = sparse(1:m, target_rows, signs, m, n);
endfunction

# Matrices
% matrices_file = {'ck400', 'fs_680_1', 'mcca', 'pde900'};
% matrices_file = {'poisson3Da', 'shermanACb'};
% matrices_file = {'poisson3Da', 'airfoil_2d'};
% matrices_file = {'e40r0100'};
# matrices_file = {'kim1'};
% matrices_file = {'cavity26'};
matrices_file = {'poisson3Da', 'poisson3Db'};

# --- Layout Parameters ---
rows = 1;
cols = 2;
hmargin = 0.08; % Left/Right margin of the figure
vmargin = 0.10; % Top/Bottom margin of the figure
hgap = 0.06;    % Horizontal gap between subplots
vgap = 0.12;    % Vertical gap between subplots (more for titles/labels)

# Calculate width and height of each individual plot box
plot_box_width = (1 - 2*hmargin - (cols-1)*hgap) / cols;
plot_box_heigth = (1 - 2*vmargin - (rows-1)*vgap) / rows;

figure('name', 'Residual / Iterations for varying matrix A');
hold on
for j = 1:length(matrices_file)

    printf("Computing matrix A: %s\n", char(matrices_file(j)));

# 1. Calculate Grid Position
    row_idx = ceil(j / cols); % 1 or 2
    col_idx = mod(j-1, cols) + 1; % 1, 2, or 3

    x_pos = hmargin + (col_idx-1)*(plot_box_width + hgap);
    y_pos = 1 - vmargin - row_idx*plot_box_heigth - (row_idx-1)*vgap;

# 2. Create axes with the manual position
    ax = subplot('Position', [x_pos, y_pos, plot_box_width, plot_box_heigth]);

    A_struct = load(strcat('sparse_matrices/',char(matrices_file(j)),'.mat'));
    A = A_struct.Problem.A;
    n = size(A,1);
    printf("Size of A; %d\n", n);

    x_true = ones(n,1);

    b = A * x_true;

    % maxit = round(n * 0.02);
    maxit = 500;

# 1. GMRES
    tt = tic();
    [~, res0] = my_gmres(A, b, 1e-6, maxit);
    tt = toc(tt);
    fprintf('GMRES: %f secs\n', tt);


# Sketching Dimensions (Adjusted for stability)
    k_gauss = 2 * maxit + 1;
    k_large = 6 * maxit + 1;
    printf("k_gauss: %d\nk_large: %d\n", k_gauss, k_large);

# 2. Gaussian
    tt = tic();
    S1 = randn(n, k_gauss) / sqrt(n);
    [~, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);
    tt = toc(tt);
    fprintf('Gaussian: %f secs\n', tt);

% # 3. Hadamard
%     S2 = hadamard(n) / sqrt(k_large);
%     S2 = S2(:, randsample(n, k_large));
%     [~, res2] = my_gmres_sketch(A, b, 1e-6, maxit, S2);

# 4. Rademacher
    tt = tic();
    S3 = (rand(n, k_gauss) > 0.5) * 2 - 1;
    [~, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);
    tt = toc(tt);
    fprintf('Rademacher: %f secs\n', tt);


# 5. SRFT (Fourier)
    tt = tic();
    I4 = randsample(n, k_large);
    S4 = @(v) select_indices(fft(v), I4);
    [~, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);
    tt = toc(tt);
    fprintf('Fourier: %f secs\n', tt);

# 6. CountSketch
    tt = tic();
    S5 = count_sketch(n, k_large);
    [~, res5] = my_gmres_sketch(A, b, 1e-6, maxit, S5);
    tt = toc(tt);
    fprintf('CountSketch: %f secs\n', tt);

# --- Subplot Rendering ---
    # subplot(2, 3, j);
    semilogy(res0, 'k--', 'LineWidth', 2); hold on;
    semilogy(res1);
    % semilogy(res2);
    semilogy(res3);
    semilogy(res4);
    semilogy(res5);

    grid on;
    title(['Matrix = ', char(matrices_file(j)), ', n = ', num2str(n)]);
# Clean up labels to save space
    if col_idx == 1
        ylabel('Residual');
    end
    if row_idx == rows
        xlabel('Iterations');
    end

# Place legend only on the first plot
    if j == 1
        legend('GMRES', 'Gaussian', 'Rademacher', 'SRFT', 'CountSketch', ...
               'Location', 'southwest', 'FontSize', 8);
    end

# Optional: Force Octave to trim excess white space inside the axes
    set(ax, 'LooseInset', get(ax, 'TightInset'));
endfor


# --- The sgtitle Workaround ---
# h = axes('visible', 'off', 'title', 'Residual vs Iterations for different Sketching Dimensions');
hold off
# set(get(h, 'title'), 'visible', 'on');
