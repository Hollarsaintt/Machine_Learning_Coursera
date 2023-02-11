function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

regTheta = theta; regTheta(1) = 0; % so that bias theta will not be regularized or penalized by regulurization.

J = 1/(2*m) * (sum(((X*theta) - y) .^ 2) + (lambda* sum((regTheta.^2)))); % Regularized linear reg cost function.


grad = 1/m * (((X' * ((X*theta) - y))) + (lambda * regTheta)); % regularized gradient of J wrt theta
								% regTheta takes care of not regularizing the bias theta term.








% =========================================================================

grad = grad(:); %changes grad vector to column vector.

end
