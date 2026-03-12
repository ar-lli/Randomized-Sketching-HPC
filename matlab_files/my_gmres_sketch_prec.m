function [x, res] = my_gmres_sketch_prec(A, b, tol, maxit, S, M_inv)
% Preconditioned Sketched GMRES
% A: Matrix or function handle
% b: Right-hand side vector
% tol: Tolerance
% maxit: Maximum iterations
% S: Sketching matrix or function handle
% M_inv: Right Preconditioner function handle (M_inv(v) = M \ v)

% Handle Sketching operator
if ~isa(S, 'function_handle')
    S_func = @(x) S' * x;
else
    S_func = S;
end

% Handle Preconditioner (default to identity)
if nargin < 6 || isempty(M_inv)
    M_inv = @(x) x;
end

Sb = S_func(b);
nrmSb = norm(Sb);
V = b / nrmSb;

if ismatrix(A)
    A_func = @(x) A * x;
else
    A_func = A;
end

% Pre-allocate for performance
SAV = zeros(length(Sb), maxit);
H = zeros(maxit + 1, maxit); 

for j = 1 : maxit
    % --- RIGHT PRECONDITIONING STEP ---
    % Instead of A(v), we use A(M_inv(v))
    z = M_inv(V(:,j));
    w = A_func(z);
    % ----------------------------------
    
    % Store the sketched action: S * A * M_inv * V_j
    SAV(:,j) = S_func(w);

    % Standard Arnoldi process (on the unsketched w)
    for i = 1 : j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);
    V(:,j+1) = w / H(j+1,j);

    % Solve the sketched least squares problem
    y = SAV(:,1:j) \ Sb;
    
    % Calculate relative residual
    % Note: x = M_inv * V * y
    x_curr = M_inv(V(:, 1:j) * y);
    res(j) = norm(A_func(x_curr) - b) / norm(b);

    if res(j) < tol
        break; 
    end
end

% Final solution recovery
x = M_inv(V(:, 1:j) * y); 

end