function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

pMat = sigmoid(X * all_theta'); % fully vectorized form of using the learned parameters to predict the class of each...
					%training example(x) in X. pMat returns the matrix of all predictions for each example...
					% where the rows are each predictions for each example and each column is the 
					% predicted probability h(x) that x belongs to class k (k is indices of the columns)
					
[maxP,i] = max(pMat, [], 2); % find the index where the probability that it belongs to the class is the highest...
	 % that is the class the training example x belong; return both the max predicted probabilities and their corresponding...
	 % indices rowWise.
	 
p = i; % set the class each example belong, to p, since the class it belongs is same as the index of the max...
       % predicted probabilities of the classifiers.










% =========================================================================


end
