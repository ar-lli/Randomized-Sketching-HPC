function [x, res] = my_gmres(A, b, tol, maxit)

nrmb = norm(b);
V = b / nrmb;

if ismatrix(A)
    A = @(x) A * x;
end

for j = 1 : maxit
    w = A(V(:,j));

    for i = 1 : j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);

    V(:,j+1) = w / H(j+1,j);

    % Solve the least squares problems
    y = H \ (eye(j+1, 1) * nrmb);

    res(j) = norm(H * y - eye(j+1, 1) * nrmb) / nrmb;

    if res(j) < tol
        break; % Exit if the residual is below the tolerance
    end
end

x = V(:, 1:j) * y; % Update the solution vectors

end