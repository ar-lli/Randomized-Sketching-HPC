function [x, res] = my_gmres(A, b, tol, maxit)
    nrmb = norm(b);
    V = b / nrmb;

    if ismatrix(A)
        A_func = @(x) A * x;
    else
        A_func = A;
    end

    H = []; % Initialize H
    for j = 1 : maxit
        w = A_func(V(:,j));

        for i = 1 : j
            H(i,j) = V(:,i)' * w;
            w = w - H(i,j) * V(:,i);
        end
        H(j+1,j) = norm(w);
        V(:,j+1) = w / H(j+1,j);

        % Solve the least squares problems
        e1 = zeros(j+1, 1); e1(1) = 1;
        y = H \ (e1 * nrmb);

        % Relative residual of the "real" solution
        res(j) = norm(A_func(V(:,1:j)*y) - b) / nrmb; 

        if res(j) < tol
            break; 
        end
    end
    x = V(:, 1:j) * y; 
end