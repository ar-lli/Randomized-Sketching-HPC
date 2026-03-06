1;

# Functions definitions
function y = select_indices(x,I)
    y = x(I);
endfunction

function ST = count_sketch(m, n)
# m: original (large) dimension
# n: reduced (target) dimension
    
# Each of the m original rows must map to one of the n target buckets
    target_rows = randi(n, m, 1);
    
# Generate random signs for each of the m original rows
    signs = 2 * (rand(m, 1) > 0.5) - 1;
    
# Create the sparse matrix ST (m x n)
# Each row i of ST has a single +/-1 at column target_rows(i)
    ST = sparse(1:m, target_rows, signs, m, n);
end


# 2d Laplace

n = 128; % Condition number goes as n^2
t = linspace(0, 1, n + 2);
[xx, yy] = meshgrid(t, t);
X = -sin(xx .* pi) .* exp(yy) .* (yy.^2 - yy);
x = X(2:end-1, 2:end-1);
x = x(:);

h = 1 / (n + 1);
A = gallery('poisson', n) / h^2;

# Optional: convection term to make everything non-symmetric
eta = 0.1;
D = spdiags(ones(n^2, 1) * [-1 1], [-1, 1], n^2, n^2) / (2 * h);

A = A + eta * D;
b = A * x;

maxit = 300;

#################################################################################
######### Grid plot as the sketching size varies: Iterations / Residuals ########
#################################################################################


# Solve with GMRES
tt0 = tic;
[x0, res0] = my_gmres(A, b, 1e-6, maxit);
tt0 = toc(tt0);
fprintf('GMRES: %f secs\n', tt0);

# --- Plot Layout Parameters ---
rows = 2;
cols = 3;
hmargin = 0.08; % Left/Right margin of the figure
vmargin = 0.10; % Top/Bottom margin of the figure
hgap = 0.06;    % Horizontal gap between subplots
vgap = 0.12;    % Vertical gap between subplots (more for titles/labels)

# Calculate width and height of each individual plot box
w = (1 - 2*hmargin - (cols-1)*hgap) / cols;
h = (1 - 2*vmargin - (rows-1)*vgap) / rows;
k_times_interval = 1:6;

figure('name', 'Residual / Iterations for varying sketching dimension (s)');
hold on

# Loop on sketching dimension
for i = 1:length(k_times_interval)

# 1. Calculate Grid Position
    row_idx = ceil(i / cols); % 1 or 2
    col_idx = mod(i-1, cols) + 1; % 1, 2, or 3
    
    x_pos = hmargin + (col_idx-1)*(w + hgap);
    y_pos = 1 - vmargin - row_idx*h - (row_idx-1)*vgap;
    
# 2. Create axes with the manual position
    ax = subplot('Position', [x_pos, y_pos, w, h]);

#
    k_val = k_times_interval(i);
    
    # Calculate sketching sizes
    k1 = k_val * maxit + 1;
    
    # Run solvers and capture full residual histories
    # Gaussian
    S1 = randn(n^2, k1) / n;
    [~, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);

    # Hadamard
    S2 = hadamard(n^2) / sqrt(k1);
    S2 = S2(:, randsample(n^2, k1));
    [~, res2] = my_gmres_sketch(A, b, 1e-6, maxit, S2);

    # Rademacher
    S3 = (rand(n^2, k1) > .5) * 2 - 1;
    [~, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);

    # SRFT
    I4 = randsample(n^2, k1);
    S4 = @(x) select_indices(fft(x), I4);
    [~, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);

    # CountSketch Sparse sketching
    S5 = count_sketch(n^2,k1);
    [~, res5] = my_gmres_sketch(A, b, 1e-6, maxit, S5);


    # --- Plotting in the Grid ---
    # subplot(2, 3, i); % Creates a 2x3 grid
    hold on
    semilogy(res0, 'k--', 'LineWidth', 1.5); hold on; # Original GMRES for reference
    semilogy(res1);
    semilogy(res2);
    semilogy(res3);
    semilogy(res4);
    semilogy(res5);
    
		grid on;
    title(['Sketching Dim s = ', num2str(k_val), '* maxit + 1']);

    if col_idx == 1
        ylabel('Residual');
    end
    if row_idx == rows
        xlabel('Iterations');
    end

# Place legend only on the first plot
    if i == 1
        legend('GMRES', 'Gaussian', 'Hadamard', 'Rademacher', 'SRFT', 'CountSketch', ...
               'Location', 'southwest', 'FontSize', 8);
    endif
    grid on;
endfor

#
# h = axes('visible', 'off', 'title', 'Residual / Iterations for different Sketching Dimensions');
hold off
# set(get(h, 'title'), 'visible', 'on');
