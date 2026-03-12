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

n = 32;
t = linspace(0,1,n+2);
[xx,yy] = meshgrid(t,t);

X = -sin(xx.*pi).*exp(yy).*(yy.^2-yy);
x = X(2:end-1,2:end-1);
x = x(:);

h = 1/(n+1);
A = gallery('poisson',n)/h^2;

eta = 0.1;
D = spdiags(ones(n^2,1)*[-1 1],[-1,1],n^2,n^2)/(2*h);

A = A + eta*D;
b = A*x;

maxit = 10;

%% -------- Parameters --------

rows = 2;
cols = 3;
k_times = 1:6;

methods = {"GMRES","Gaussian","Hadamard","DCT","Rademacher","SRFT","CountSketch"};
nm = length(methods);
times = zeros(length(k_times), nm);

fig1 = figure('name','Residual vs Iterations');

%% -------- Reference GMRES --------
% We run this once and copy the time for all sketch size rows for comparison
tt_ref = tic();
[~,res{1}] = my_gmres(A,b,1e-6,maxit);
time_ref = toc(tt_ref);
fprintf("GMRES: %.3f sec\n", time_ref);

%% -------- Main Loop --------

for j = 1:length(k_times)

    k = k_times(j);
    s = k*maxit + 1;
    times(j,1) = time_ref; % Constant reference time

    fprintf("\nSketch size s = %d\n",s);

    % Define Sketches
    S_gauss = randn(n^2,s)/sqrt(n^2);
    S_hadamard = srht_sketch(n^2,s);
    S_dct = sdct_sketch(n^2,s);
    S_radem = (rand(n^2,s)>0.5)*2-1;
    idx = randsample(n^2,s);
    S_srft = @(v) select_indices(fft(v),idx);
    S_count = count_sketch(n^2,s);

    sketches = {S_gauss, S_hadamard, S_dct, S_radem, S_srft, S_count};

    for m = 2:nm
        tt = tic();
        [~,res{m}] = my_gmres_sketch(A,b,1e-6,maxit,sketches{m-1});
        times(j,m) = toc(tt);
        fprintf('%s: %.3f sec\n', methods{m}, times(j,m));
    end

    %% -------- Plot Convergence --------
    figure(fig1);
    subplot(rows,cols,j)
    hold on
    styles = {"k--","b","r","g","m","c","y"};

    for m = 1:nm
        semilogy(res{m},styles{m},'LineWidth',1.5);
    end

    grid on
    title(sprintf("s = %d*maxit + 1",k))
    if j == 1, legend(methods,'Location','southwest','FontSize',8); end
    if mod(j-1,cols)==0, ylabel("Residual"); end
    if j > (rows-1)*cols, xlabel("Iterations"); end
end

% Save the convergence plot
print(fig1, './plots/test_k_2d.png', '-dpng');

%% -------- Plot Runtime Scaling --------
fig2 = figure('name','Runtime Scaling');
hold on
colors = lines(nm);
s_values = k_times * maxit + 1;

for m = 1:nm
    plot(s_values, times(:,m), '-o', ...
        'Color', colors(m,:), ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6);
end

grid on
xlabel('Sketch size (s)')
ylabel('Execution time (seconds)')
title('Runtime scaling vs Sketch Size')
legend(methods,'Location','northwest')
hold off

% Save the runtime plot
print(fig2, './plots/test_k_2d_runtime.png', '-dpng');

%% -------- Timing Summary --------
disp("Execution times (seconds) for different sketch sizes:")
disp(array2table(times,"VariableNames",methods,"RowNames",string(s_values)))
% Create the table object
T = array2table(times, "VariableNames", methods, "RowNames", string(s_values));
% writetable(T, './csv/test_k_2d_runtime.csv');
% disp("Table saved to test_k_2d_runtime.csv");