function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

covSigma = 1/m*(X'*X); % vectorized computation of the covariance matrix sigma which is nxn dimensional

[U S] = svd(covSigma); % the svd function return U matrix and S matrix which contains the eigenVectors and eigen values...
			% which can be used in data compression or dimensionality reduction.




% =========================================================================

end
