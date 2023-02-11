function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    pred = zeros(size(pval));  % initialize prediction to be zeros for all the cv data.
    pred(find(pval < epsilon)) = 1; % set where p(x) is less than epsilon to be anomaly(1) otherwise non-anomaly(0)...
    					% which is already the defaulted value.
    
    tp = sum((pred == 1) & (yval == 1)); % number of true positive
    fp = sum((pred == 1) & (yval == 0)); % number of false positive
    fn = sum((pred == 0) & (yval == 1)); % number of false negative
    
    prec = tp / (tp + fp); % calculate the precision using the precision known formulae
    rec = tp / (tp + fn);  % calculate the recall using the recall known formulae
    
    F1 = 2 * prec * rec / (prec + rec); % compute the f1-score using the known formulae










    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
