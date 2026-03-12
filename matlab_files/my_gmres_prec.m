function [x, res] = my_gmres_prec(A, b, tol, maxit, M_inv)
    nrmb = norm(b);
    V = b / nrmb;

    % Handle A as matrix or function handle
    if ismatrix(A)
        A_func = @(x) A * x;
    else
        A_func = A;
    end
    
    % Default to Identity if no preconditioner is provided
    if nargin < 5 || isempty(M_inv)
        M_inv = @(x) x; 
    end

    H = []; 
    for j = 1 : maxit
        % --- RIGHT PRECONDITIONING STEP ---
        % We apply the inverse of M to the basis vector
        z = M_inv(V(:,j)); 
        w = A_func(z);
        % ----------------------------------

        for i = 1 : j
            H(i,j) = V(:,i)' * w;
            w = w - H(i,j) * V(:,i);
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w / H(j+1,j);

        % Solve the small projected least squares problem
        e1 = zeros(j+1, 1); e1(1) = 1;
        y = H \ (e1 * nrmb);

        % Update solution and calculate relative residual
        x_curr = M_inv(V(:, 1:j) * y); 
        res(j) = norm(A_func(x_curr) - b) / nrmb; 

        if res(j) < tol
            break; 
        end
    end
    % Final solution recovery
    x = M_inv(V(:, 1:j) * y); 
end