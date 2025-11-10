function hat_X = solve_hat_X(X, W, b, Y, M, gamma, n1, L1)
two_n = 2 * n1;
A = W * W';
P = gamma * (L1 - M')' * (L1 - M);
B = (Y - X * W - two_n * b') * W';
hat_X = sylvester(P, A, B);
end
