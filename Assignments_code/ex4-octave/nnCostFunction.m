function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%forward prop for three layers architectures.
a1 = [ones(1,size(X',2)); X']; % s1+1 x m. add the bias input to X and make each training example be column.
				% Note: sI means number of units in layer I.
a2 = sigmoid(Theta1*a1); % s2 x m
a2 = [ones(1,size(a2,2)); a2]; % s2+1 x m.  % add the bias input before using it as input to compute a3.
a3 = sigmoid(Theta2*a2); % s3 x m
h = a3'; % m x s3. makes hypothesis h agrees with y, i.e training examples in row and the class logical array in column.


% cost function J(theta) with regularization.
yInit = zeros(m,num_labels);  % initialize y as a logical array to all 0s.

for(i=1:m),
	yInit(i,y(i)) = 1; % change yInit to logical arrays representing the class labels for each training example.
end;
y = yInit; % m x s3. % change our original y to logical array representing the classes for each example.

costTerm = ((y.*log(h)) + ((1-y) .* log(1-h))); % hold the cost term for all class prediction for all training examples.
						 % as a matrix grid.
jNoReg = (-1/m) * (sum(sum(costTerm, 2))); % compute the cost without regularization. Note: 2 symbolizes summing it rowwise(it is
						% not important or necessary) just there for consistency.

regTerm = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end) .^2))) + sum(sum((Theta2(:,2:end) .^ 2)))); % regularization of the parameters
											% no penalization of the first column...
											% which always hold the bias term of the parameters.
											
J = jNoReg + regTerm; % success.

% Back prop: compute the derivative of J wrt all its parameters.

dAll2 = zeros(num_labels, hidden_layer_size + 1); % initializes accumulator for derivative wrt Theta2.
dAll1 = zeros(hidden_layer_size, input_layer_size + 1); % initializes accumulator for derivative wrt Theta1.

for (t=1:m),
	delta3 = h(t,:) - y(t,:); % Error in layer 3 for the t.  1 x s3
	delta3 = delta3';  % changed to s3 x 1 for consistency and appropriacy.
	delta2 = ((Theta2(:,2:end)' * delta3) .* (sigmoidGradient(Theta1*a1(:,t)))); % since a1 is s1+1 x m ...
										 % means each column of a1 is a training example a1. 
										 % s2 x 1

	dAll2 = dAll2 + (delta3 * (a2(:,t))'); % derivative of J wrt Theta2 parameters correspondingly.
					% a2 is s2+1 x m, means each column is the activation of layer 2 ...
					% for each training example and the bias unit has been added(bias unit=1).
					% s3 x s2+1
	
					
	dAll1 = dAll1 + (delta2 * (a1(:,t))'); % derivative of J wrt Theta1 parameters correspondingly.
						% a1 is s1+1 x m, means each column is the activation of layer 1...
						% for each training example and the bias unit has been added(bias unit = 1)
						% s2 x s1+1
end;

% derivative wrt to Theta1
D1RegTerms = 1/m * (dAll1(:,2:end) + (lambda * Theta1(:,2:end))); % finding the average derivative wrt Theta1 of all training examples
							% of the Non bias terms
							% and adding regularization to the Non bias terms
							
D1NoReg = 1/m * (dAll1(:,1)); % finding the average derivative of the bias terms(No regularization)
Theta1_grad = [D1NoReg D1RegTerms];

% derivative wrt to Theta2
D2RegTerms = 1/m * (dAll2(:,2:end) + (lambda * Theta2(:,2:end))); % finding the average derivative wrt Theta2 of all training examples
							% of the Non bias terms
							% and adding regularization to the Non bias terms
							
D2NoReg = 1/m * (dAll2(:,1)); % finding the average derivative of the bias terms(No regularization)
Theta2_grad = [D2NoReg D2RegTerms];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
