function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 3;
sigma = 0.1;

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

values = [0.01 0.03 0.1 0.3 1 3 10 30];
all_predictions = zeros(size(values, 2).*size(values, 2), 3);
row_store = 1;


for i = 1:size(values, 2),
	for j = 1:size(values, 2),
		C = values(i);
		sigma = values(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 		
		predictions = svmPredict(model, Xval);
		accuracy = mean(double(predictions ~= yval));
		all_predictions(row_store, 1) = C;
		all_predictions(row_store, 2) = sigma;
		all_predictions(row_store, 3) = accuracy;
		row_store += 1;
	end
end

[max_values indices] = min(all_predictions);
max_accuracy_index = indices(1, 3);
C = all_predictions(max_accuracy_index, 1)
sigma = all_predictions(max_accuracy_index, 2)
all_predictions
% =========================================================================

end
