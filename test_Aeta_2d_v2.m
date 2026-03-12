1;

pkg load statistics
pkg load signal

%% -------- Utilities --------

function y = select_indices(x,I)
    y = x(I);
endfunction

function S = sdct_sketch(n,k)
    d = sign(randn(n,1));
    idx = randperm(n,k);
    scale = sqrt(n/k);
    S = @(x) scale * dct(d .* x)(idx);
endfunction

function S = srht_sketch(n,k)
    m = 2^nextpow2(n);
    d = sign(randn(m,1));
    idx = randperm(m,k);
    scale = sqrt(m/k);
    S = @(x) scale * fwht_apply(d,idx,x,n,m);
endfunction

function y = fwht_apply(d,idx,x,n,m)
    xp = zeros(m,1);
    xp(1:n) = x;
    xp = d .* xp;
    xp = fwht(xp);
    y = xp(idx);
endfunction

function S = count_sketch(m,n)
    target_rows = randi(n,m,1);
    signs = 2*(rand(m,1)>0.5)-1;
    S = sparse(1:m,target_rows,signs,m,n);
endfunction


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

methods = {"GMRES","Gaussian","Hadamard","DCT","Rademacher","SRFT","CountSketch"};
nm = length(methods);
times = zeros(length(eta_values), nm);

fig1 = figure('name','Residual vs Iterations');


%% -------- Main Loop --------

for j = 1:length(eta_values)

    eta = eta_values(j);
    A = A_poisson + eta*D;
    b = A*x_true;

    printf("Case eta = %.2f\n",eta);

    k_gauss = 2*maxit + 1;
    k_large = 8*maxit + 1;

    %% -------- GMRES --------
    tt = tic();
    [~,res{1}] = my_gmres(A,b,1e-6,maxit);
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
        [~,res{m}] = my_gmres_sketch(A,b,1e-6,maxit,sketches{m-1});
        times(j,m) = toc(tt);
    end

    %% -------- Plot Convergence --------
    figure(fig1);
    subplot(rows,cols,j)
    hold on

    styles = {"k--","b","r","g","m","c","y"};
    for m = 1:nm
        semilogy(res{m},styles{m},'LineWidth',1.6);
    end

    grid on
    title(sprintf("\\eta = %.2f",eta))

    if j == 1
        legend(methods,'Location','southwest','FontSize',8)
    end
    if mod(j-1,cols)==0
        ylabel("Residual")
    end
    if j > (rows-1)*cols
        xlabel("Iterations")
    end
end

% Save Convergence Plot
print(fig1, './plots/test_Aeta_2d.png', '-dpng');

%% -------- Plot Runtime Scaling --------

fig2 = figure('name','Runtime Scaling vs Eta');
hold on
colors = lines(nm);

for m = 1:nm
    plot(eta_values, times(:,m), '-o', ...
        'Color', colors(m,:), ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6);
end

grid on
xlabel('\eta (convection strength)')
ylabel('Execution time (seconds)')
title('Runtime scaling as system becomes non-symmetric')
legend(methods,'Location','northwest')
hold off

% Save Runtime Plot
print(fig2, './plots/test_Aeta_2d_runtime.png', '-dpng');

%% -------- Save Results to Table and File --------

% Create a table from the timing data
T = array2table(times, "VariableNames", methods, "RowNames", string(eta_values));

% Display the table in the command window
disp("Execution times (seconds):");
disp(T);

% Save the table to a CSV file (MATLAB syntax)
writetable(T, './csv/test_Aeta_2d_runtime.csv');

printf("\nConvergence plot saved as 'convergence_eta_study.png'\n");
printf("Runtime plot saved as 'runtime_eta_study.png'\n");
printf("Timing data saved as 'timing_eta_results.csv'\n");