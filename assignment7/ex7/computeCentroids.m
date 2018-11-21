function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

myMatrix = zeros(K*2, m);
length1 = 0;
length2 = 0;
length3 = 0;

for i=1:size(X, 1)
	if idx(i) == 1
		myMatrix(i, 1:2) = X(i, :);
		length1 += 1;
	elseif idx(i) == 2
		myMatrix(i, 3:4) = X(i, :);
		length2 += 1;
	elseif idx(i) == 3
		myMatrix(i, 5:6) = X(i, :);
		length3 += 1;
	end
end

centroids(1, :) = (1/length1)*sum(myMatrix(:, 1:2));
centroids(2, :) = (1/length2)*sum(myMatrix(:, 3:4));
centroids(3, :) = (1/length3)*sum(myMatrix(:, 5:6));






% =============================================================


end

