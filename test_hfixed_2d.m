##### This test compares the iterations/residual plots of the methods as the dimension of matrix A increases.
##### Specifically, the matrix dimensions vary according to the array n_dimensions = [128, 256, 512, 1024, 2048, 4096];

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

# Matrix A dimensions
n_dimensions = [128, 256];

maxit = 500;

# --- Layout Parameters ---
rows = 2;
cols = 3;
hmargin = 0.08; % Left/Right margin of the figure
vmargin = 0.10; % Top/Bottom margin of the figure
hgap = 0.06;    % Horizontal gap between subplots
vgap = 0.12;    % Vertical gap between subplots (more for titles/labels)

# Calculate width and height of each individual plot box
plot_box_width = (1 - 2*hmargin - (cols-1)*hgap) / cols;
plot_box_heigth = (1 - 2*vmargin - (rows-1)*vgap) / rows;

figure('name', 'residual vs iterations for varying matrix A dimension (n)');
hold on
for j = 1:length(n_dimensions)

printf("Case n = %d\n\n", n_dimensions(j));
# 1. Calculate Grid Position
    row_idx = ceil(j / cols); % 1 or 2
    col_idx = mod(j-1, cols) + 1; % 1, 2, or 3
    
    x_pos = hmargin + (col_idx-1)*(plot_box_width + hgap);
    y_pos = 1 - vmargin - row_idx*plot_box_heigth - (row_idx-1)*vgap;
    
# 2. Create axes with the manual position
    ax = subplot('Position', [x_pos, y_pos, plot_box_width, plot_box_heigth]);

    n = n_dimensions(j);

    % h = 1 / (n + 1);
    h = 0.001;
    t = linspace(0, 1, n + 2);
    [xx, yy] = meshgrid(t, t);

    # True solution for generating b
    X_true = -sin(xx .* pi) .* exp(yy) .* (yy.^2 - yy);
    x_true = X_true(2:end-1, 2:end-1);
    x_true = x_true(:);

    # Base Poisson Matrix
    A_poisson = gallery('poisson', n) / h^2;
    D = spdiags(ones(n^2, 1) * [-1 1], [-1, 1], n^2, n^2) / (2 * h);

    # Define the range of eta to test
    eta = 0.5; 

    A = A_poisson + eta * D;
    b = A * x_true;
    
# 1. GMRES
    tt = tic();
    [~, res0] = my_gmres(A, b, 1e-6, maxit);
    printf("0. Computing standard GMRES...\n")
    tt = toc(tt);
    fprintf('GMRES: %f secs\n', tt);
    
# Sketching Dimensions (Adjusted for stability)
    k_gauss = 2 * maxit + 1;
    k_large = 8 * maxit + 1; 

# 2. Gaussian
    S1 = randn(n^2, k_gauss) / sqrt(n^2);
    [~, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);
    printf("1. Computing Gaussian Sketching...\n")

% # 3. Hadamard
%     S2 = hadamard(n^2) / sqrt(k_large);
%     S2 = S2(:, randsample(n^2, k_large));
%     [~, res2] = my_gmres_sketch(A, b, 1e-6, maxit, S2);
%     printf("2. Computing Hadamard Sketching...\n")

# 4. Rademacher
    S3 = (rand(n^2, k_gauss) > 0.5) * 2 - 1;
    [~, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);
    printf("3. Computing Rademacher Sketching...\n")

# 5. SRFT (Fourier)
    I4 = randsample(n^2, k_large);
    S4 = @(v) select_indices(fft(v), I4);
    [~, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);
    printf("4. Computing Fourier Sketching...\n")

# 6. CountSketch Sparse
    S5 = count_sketch(n^2, k_large);
    [~, res5] = my_gmres_sketch(A, b, 1e-6, maxit, S5);
    printf("5. Computing CountSketch Sparse Sketching...\n")

# --- Subplot Rendering ---
    # subplot(2, 3, j);
    semilogy(res0, 'k--', 'LineWidth', 2); hold on;
    semilogy(res1);
    % semilogy(res2);
    semilogy(res3);
    semilogy(res4);
    semilogy(res5);
    
    grid on;
    title(['n = ', num2str(n_dimensions(j))]);
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
