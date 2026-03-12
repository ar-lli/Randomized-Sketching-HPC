1;

# Functions definitions
function y = select_indices(x,I)
    y = x(I);
endfunction

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

#######################################################################
###################### 1 - Iterations / Residual ######################
#######################################################################

# Solve with GMRES
tt = tic;
[x0, res0] = my_gmres(A, b, 1e-6, maxit);
tt = toc(tt);
fprintf('GMRES: %f secs\n', tt);

# Gaussian
tt = tic;
k1 = 2 * maxit + 1;
S1 = randn(n^2, k1) / n;
[x1, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);
tt = toc(tt);
fprintf('Gaussian: %f secs\n', tt);

# Hadamard
tt = tic;
k2 = 8 * maxit + 1;
S2 = hadamard(n^2) / sqrt(k2);
S2 = S2(:, randsample(n^2, k2));
#I2 = randsample(n^2, k2);
#S2 = @(x) select_indices(fwht(log2(n)*2, x), I2);
#nn = n^2 / 4;
# ln = log(n^2) - 2;
# S2 = @(x) select_indices([ fwht(ln, x(1:nn)) ; fwht(ln, x(nn+1:2*nn)) ; fwht(ln, x(2*nn+1:3*nn)) ; fwht(ln,x(3*nn+1:4*nn)) ], I2);
[x2, res2] = my_gmres_sketch(A, b, 1e-6, maxit, S2);
tt = toc(tt);
fprintf('FWHT: %f secs\n', tt);

# Rademacher: +/- 1 con probabilità 1/2
tt = tic;
k3 = 2 * maxit + 1;
S3 = (rand(n^2, k3) > .5) * 2 - 1;
[x3, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);
tt = toc(tt);
fprintf('Rademacher: %f secs\n', tt);

# Subsampled Fourier Transform (volendo da sostituire con DCT / DST)
tt = tic;
k4 = 8 * maxit + 1;
I4 = randsample(n^2, k4);
S4 = @(x) select_indices(fft(x), I4);
#S4 = dftmtx(n^2);
#S4 = S4(:, randsample(n^2, k4));
[x4, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);
tt = toc(tt);
fprintf('SRFT: %f secs\n', tt);

#######################################################################
############# 2 - k Magnitude (sketching dim.) / Residual #############
############ 3 - k Magnitude (sketching dim.) / Exe. time #############
#######################################################################

# Solve with GMRES
tt0 = tic;
[x0, res0] = my_gmres(A, b, 1e-6, maxit);
tt0 = toc(tt0);
fprintf('GMRES: %f secs\n', tt);

k_times_interval = 1:5;

res_k = zeros(4,length(k_times_interval));
tt_k = zeros(4,length(k_times_interval));

# Loop on sketching dimension
for k_times = k_times_interval

	k_times
# Gaussian
	tt = tic;
	k1 = k_times * maxit + 1;
	S1 = randn(n^2, k1) / n;
	[x1, res1] = my_gmres_sketch(A, b, 1e-6, maxit, S1);
	res_k(1, k_times) = res1(end);
	tt_k(1,k_times) = toc(tt);

# Hadamard
	tt = tic;
	k2 = k_times * maxit + 1;
	S2 = hadamard(n^2) / sqrt(k2);
	S2 = S2(:, randsample(n^2, k2));
	[x2, res2] = my_gmres_sketch(A, b, 1e-6, maxit, S2);
	res_k(2, k_times) = res2(end);
	tt_k(2,k_times) = toc(tt);

# Rademacher: +/- 1 con probabilità 1/2
	tt = tic;
	k3 = k_times * maxit + 1;
	S3 = (rand(n^2, k3) > .5) * 2 - 1;
	[x3, res3] = my_gmres_sketch(A, b, 1e-6, maxit, S3);
	res_k(3, k_times) = res3(end);
	tt_k(3,k_times) = toc(tt);

# Subsampled Fourier Transform (volendo da sostituire con DCT / DST)
	tt = tic;
	k4 = k_times * maxit + 1;
	I4 = randsample(n^2, k4);
	S4 = @(x) select_indices(fft(x), I4);
	[x4, res4] = my_gmres_sketch(A, b, 1e-6, maxit, S4);
	res_k(4, k_times) = res4(end);
	tt_k(4,k_times) = toc(tt);

endfor


figure;
# Iteration / Residual
subplot(1,3,1);
hold on;
title('Residual plot');
semilogy(res0); 
semilogy(res1);
semilogy(res2);
semilogy(res3)
semilogy(res4)
legend('GMRES', 'Gaussian', 'Hadamard', 'Rademacher', 'SRFT');
# Sketching dimension / Residual
subplot(1,3,2);
hold on;
title('Residual wrt k where s = k*max_it + 1');
semilogy(k_times_interval, res0(end)*ones(1,length(k_times_interval)));
semilogy(k_times_interval, res_k(1,:), '-o');
semilogy(k_times_interval, res_k(2,:), '-o');
semilogy(k_times_interval, res_k(3,:), '-o');
semilogy(k_times_interval, res_k(4,:), '-o');
legend('GMRES', 'Gaussian', 'Hadamard', 'Rademacher', 'SRFT');
# Sketching dimension / Execution time
subplot(1,3,3);
hold on;
title('Runtime wrt k where s = k*max_it + 1');
plot(k_times_interval, tt0*ones(1,length(k_times_interval)));
plot(k_times_interval, tt_k(1,:));
plot(k_times_interval, tt_k(2,:));
plot(k_times_interval, tt_k(3,:));
plot(k_times_interval, tt_k(4,:));
legend('GMRES', 'Gaussian', 'Hadamard', 'Rademacher', 'SRFT');