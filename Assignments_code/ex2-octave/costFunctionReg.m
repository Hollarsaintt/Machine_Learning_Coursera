function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J_without_reg = -1/m * ((y'*log(sigmoid(X*theta))) + ((1-y)' * log(1-sigmoid(X*theta))));

J =  J_without_reg + ... 
	((1/(2*m)) * lambda * ((theta(2:end))' * theta(2:end))); %we don't penalize theta_with_zero (intercept parameter term)

grad(1) = 1/m * ((X(:,1))'* (sigmoid(X*theta) - y)); % grad wrt 1st theta term which is not penalize.

grad(2:end) = (1/m .* ((X(:, 2:end))' * (sigmoid(X*theta)-y))) ...
	.+ (lambda/m .* theta(2:end)); %grads + term due to regularization



% =============================================================

end
