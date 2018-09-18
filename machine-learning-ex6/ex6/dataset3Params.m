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

choice = [0.01 ; 0.03 ; 0.1 ; 0.3 ; 1.0 ; 3.0 ; 10 ; 30 ];
min_error_till_now = 1/0; #assigned to infinity
C_final = 0;
sigma_final = 0;
for i = [1:size(choice)], #iterator for C
  for j = [1:size(choice)], #iterator for sigma
    C = choice(i);
    sigma = choice(j);
    
    #train and predict
    model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
    predictions = svmPredict(model,Xval);
    error = mean(double(predictions ~= yval));
    if ( error < min_error_till_now )
      min_error_till_now = error;
      C_final = C;
      sigma_final = sigma;
     endif
  end
 end
C = C_final;
sigma = sigma_final;
% =========================================================================

end
