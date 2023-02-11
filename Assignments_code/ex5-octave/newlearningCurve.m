function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
    
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

%error function or cost function short hand or function handle.
costJ = @(X,y,theta,lambda)linearRegCostFunction(X,y,theta,lambda);



for i = 1:m,
	storeTrain = 0; storeVal = 0;
	for r = 1:50, % randomly select i element from X,y and Xval,yval 50 times and find their average error.
		rdTrain = randperm(size(X,1)); rdVal = randperm(size(Xval,1));
		if i <= length(yval),
			trainIndex = rdTrain(1:i); valIndex = rdVal(1:i);
		else,
			trainIndex = rdTrain(1:i); valIndex = rdVal(1:length(yval));
		end;
		theta = trainLinearReg(X(trainIndex,:), y(trainIndex), 0.01); % learn on i # of training examples.
		trainError = costJ(X(trainIndex,:), y(trainIndex), theta,0);
		storeTrain += trainError;
		valError = costJ(Xval(valIndex,:), yval(valIndex), theta,0);
		storeVal += valError;
	end;
	
	
error_train(i) = storeTrain/50; % find average Train error over 50 random computations
error_val(i) = storeVal/50; % find average cross validation error over 50 random computations
end;



% -------------------------------------------------------------

% =========================================================================
disp("Whooh!, i'm in the new learning curve...\n");
end
