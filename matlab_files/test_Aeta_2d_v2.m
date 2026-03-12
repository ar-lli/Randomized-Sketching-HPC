% MATLAB script
% Requires Statistics and Signal Processing Toolboxes

%% -------- Problem Setup --------
n = 128;
maxit = 300;
h = 1/(n+1);

t = linspace(0,1,n+2);
[xx,yy] = meshgrid(t,t);

X_true = -sin(xx.*pi).*exp(yy).*(yy.^2-yy);
x_true = X_true(2:end-1,2:end-1);
x_true = x_true(:);

A_poisson = gallery('poisson',n)/h^2;
D = spdiags(ones(n^2,1)*[-1 1],[-1,1],n^2,n^2)/(2*h);

%% -------- Parameters --------
eta_values = linspace(0.1,1,6);
rows = 2;
cols = 3;

methods = {'GMRES','Gaussian','Hadamard','DCT','Rademacher','SRFT','CountSketch'};
nm = length(methods);
times = zeros(length(eta_values), nm);

fig1 = figure('Name','Residual vs Iterations');

%% -------- Main Loop --------
for j = 1:length(eta_values)
    eta = eta_values(j);
    A = A_poisson + eta*D;
    b = A*x_true;

    fprintf('Case eta = %.2f\n', eta);

    k_gauss = 2*maxit + 1;
    k_large = 8*maxit + 1;

    % GMRES
    tt = tic();
    [~,res_cell{1}] = my_gmres(A,b,1e-6,maxit);
    times(j,1) = toc(tt);

    % Sketches
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
    end

    % Plotting
    figure(fig1);
    subplot(rows,cols,j)
    hold on
    styles = {'k--','b','r','g','m','c','y'};
    for m = 1:nm
        semilogy(res_cell{m},styles{m},'LineWidth',1.6);
    end
    grid on;
    title(sprintf('\\eta = %.2f',eta));
    if j == 1, legend(methods,'Location','southwest','FontSize',8); end
    if mod(j-1,cols)==0, ylabel('Residual'); end
    if j > (rows-1)*cols, xlabel('Iterations'); end
end

print(fig1, 'test_Aeta_2d.png', '-dpng');

% Runtime Plot
fig2 = figure('Name','Runtime Scaling vs Eta'); hold on;
colors = lines(nm);
for m = 1:nm
    plot(eta_values, times(:,m), '-o', 'Color', colors(m,:), 'LineWidth', 1.5);
end
grid on; xlabel('\eta'); ylabel('Seconds');
title('Runtime scaling'); legend(methods);
print(fig2, 'test_Aeta_2d_runtime.png', '-dpng');

% Save Results
T = array2table(times, 'VariableNames', methods, 'RowNames', string(eta_values));
writetable(T, 'test_Aeta_2d_runtime.csv');

%% -------- Utilities --------
function y = select_indices(x,I), y = x(I); end
function S = sdct_sketch(n,k)
    d = sign(randn(n,1)); idx = randperm(n,k); scale = sqrt(n/k);
    S = @(x) scale * subsref(dct(d .* x), struct('type', '()', 'subs', {{idx}}));
end
function S = srht_sketch(n,k)
    m = 2^nextpow2(n); d = sign(randn(m,1)); idx = randperm(m,k); scale = sqrt(m/k);
    S = @(x) scale * fwht_apply(d,idx,x,n,m);
end
function y = fwht_apply(d,idx,x,n,m)
    xp = zeros(m,1); xp(1:n) = x; xp = d .* xp; xp = fwht(xp); y = xp(idx);
end
function S = count_sketch(m,n)
    target_rows = randi(n,m,1); signs = 2*(rand(m,1)>0.5)-1;
    S = sparse(1:m,target_rows,signs,m,n);
end