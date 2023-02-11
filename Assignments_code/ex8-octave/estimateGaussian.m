function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = mean(X);  % find the mean of each feature in X (since each column in X is one feature)

sigma2 = 1/m * sum((bsxfun(@minus, X, mu)).^2); % find the variance(sigma^2) of each feature in X ....
					     % i.e. find variance columnwise, since each column in X is one feature.


mu = mu'; sigma2 = sigma2'; % return column vector of mu and sigma2 rather than row vector.








% =============================================================


end
