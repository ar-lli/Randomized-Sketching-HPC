function [x, res] = my_gmres_sketch(A, b, tol, maxit, S)
% MATLAB-compatible version of sketched GMRES

if ~isa(S, 'function_handle')
    % MATLAB requires the transpose to be explicit for non-function handles
    S_func = @(x) S' * x;
else
    S_func = S;
end

Sb = S_func(b);
nrmSb = norm(Sb);
V = b / nrmSb;

if ismatrix(A)
    A_func = @(x) A * x;
else
    A_func = A;
end

% Pre-allocate SAV matrix for performance
SAV = zeros(length(Sb), maxit);
H = zeros(maxit + 1, maxit); % Pre-allocate Hessenberg matrix

for j = 1 : maxit
    w = A_func(V(:,j));
    SAV(:,j) = S_func(w);

    for i = max(1, j - 1) : j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);
    V(:,j+1) = w / H(j+1,j);

    % Solve the least squares problem: min_y || (S*A*V)*y - S*b ||_2
    y = SAV(:,1:j) \ Sb;
    
    % Relative residual of the "real" solution
    res(j) = norm(A_func(V(:,1:j)*y) - b) / norm(b);

    if res(j) < tol
        break; 
    end
end

x = V(:, 1:j) * y; 

end