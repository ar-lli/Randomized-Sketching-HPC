function [x, res] = my_gmres_sketch(A, b, tol, maxit, S)

%n = size(b, 1);

%S = randn(n, 2*maxit+1) / sqrt(n);
if ~isa(S, 'function_handle')
    S = @(x) S' * x;
end

% nrmb = norm(b);
Sb = S(b);
nrmb = norm(Sb);
V = b / nrmb;

if ismatrix(A)
    A = @(x) A * x;
end

SAV = zeros(length(Sb), 0);

for j = 1 : maxit
    w = A(V(:,j));
    SAV(:,j) = S(w);

    for i = max(1, j - 1) : j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);
    V(:,j+1) = w / H(j+1,j);

    % Solve the least squares problem
    % min_y || (S*A*V)*y - S*b ||_2
    y = SAV(:,1:j) \ Sb;
    
    % res(j) = norm(SAV(:,1:j) * y - Sb) / nrmb;
    res(j) = norm(A(V(:,1:j)*y) - b) / norm(b);

    if res(j) < tol
        break; % Exit if the residual is below the tolerance
    end
end

x = V(:, 1:j) * y; % Update the solution vectors

end