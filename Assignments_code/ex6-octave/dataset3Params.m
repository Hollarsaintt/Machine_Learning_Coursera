function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% default x1 and x2 so the below function handle will work
%x1 = [0 0]; x2 = [0 0];


test = [0.01 0.03 0.1 0.3 1 3 10 30]; % holds the C and sigma parameter to cross validate.
error = 9e9; %defaulted to very high value

% loop that searches the best C and sigma on the cross validation set.
for i = 1:length(test),
	for j = 1:length(test),
		cTemp = test(i); sigmaTemp = test(j);
		model = svmTrain(X, y, cTemp, @(x1, x2) gaussianKernel(x1, x2, sigmaTemp));
		pred = svmPredict(model, Xval);
		errorVal = mean(pred ~= yval);
		if errorVal < error,
			error = errorVal; C = cTemp; sigma = sigmaTemp;
		end;
	end
end			
		

disp([C sigma error]);




% =========================================================================

end
