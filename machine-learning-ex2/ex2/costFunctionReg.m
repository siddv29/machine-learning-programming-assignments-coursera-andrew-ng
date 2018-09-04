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

h = hFunction(theta,X);
term1 = arrayfun(@log,h);
term2 = arrayfun(@log, 1.-h);
regularizationTerm = (lambda/(2*m))*theta.^2;
regularizationTerm(1) = 0;
J = (-1/m) * sum( (y'*term1 + (1-y)'*term2),2) + sum(regularizationTerm,1);

regularizationTerm = (lambda/m).*theta;
regularizationTerm(1) = 0;
grad = (1/m) * (X' * (hFunction(theta,X)-y)) + regularizationTerm;


% =============================================================

end
