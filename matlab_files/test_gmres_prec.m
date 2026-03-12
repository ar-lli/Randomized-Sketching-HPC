1;

pkg load statistics
pkg load signal

%% -------------------- Sketching Utilities --------------------

function y = select_indices(x, I)
    y = x(I);
endfunction

function S = sdct_sketch(n, k)
    d = sign(randn(n,1));
    idx = randperm(n, k);
    scale = sqrt(n/k);
    S = @(x) scale * dct(d .* x)(idx);
endfunction

function S = srht_sketch(n, k)
    m = 2^nextpow2(n);
    d = sign(randn(m,1));
    idx = randperm(m, k);
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

function S = count_sketch(m, n)
    target_rows = randi(n, m, 1);
    signs = 2 * (rand(m,1) > 0.5) - 1;
    S = sparse(1:m, target_rows, signs, m, n);
endfunction

%% -------------------- Experiment Setup --------------------

% n_dimensions = 150:50:300;
n_dimensions = [50, 70];
maxit = 50;
eta = 0.5;

%% -------------------- Main Loop --------------------

for j = 1:length(n_dimensions)

    n = n_dimensions(j);
    printf("Case n = %d\n\n", n);

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
    d = diag(A);
    M_inv = @(v) v ./ d;
    [~,res0] = my_gmres_preconditioned(A,b,1e-6,maxit);
    tt = toc(tt);
    printf('GMRES PRE: %.3f sec\n', tt);

    S = srht_sketch(n^2,k_gauss);

    tt = tic();
    [~, res1] = my_gmres_sketch_preconditioned(A,b,1e-6,maxit,S, M_inv);
    tt = toc(tt);
    printf('DCT: %.3f sec\n', tt);
endfor