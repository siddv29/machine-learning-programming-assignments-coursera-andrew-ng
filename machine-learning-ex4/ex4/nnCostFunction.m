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

s1 = input_layer_size;
s2 = hidden_layer_size;
s3 = num_labels;
J_total = 0;
for i = 1 : m
	a1 = X(i,:)';
	a2 = sigmoid(Theta1*[1;a1]);
	a3 = sigmoid(Theta2*[1;a2]);
	y_actual = zeros(s3,1);
	y_actual(y(i)+1) = 1;
	J_current = (y_actual.*log(a3)+(1-y_actual).*log(1-a3));
	J_total = J_total + sum(J_current,1);
	% all computed in forward pass itself
	del_J_del_a3 = (a3 - y_actual)./((a3).*(1-a3)) * (1/m);
	del_a3_del_a2 = ((a3).*(1-a3)).*Theta2;
	del_a3_del_w2 = ((a3).*(1-a3)).*[1;a2]';

	del_a2_del_a1 = ((a2).*(1-a2)).*Theta1;
	del_a2_del_w1 = ((a2).*(1-a2)).*[1;a1]';
	
	%backward pass
	%layer 3
	Theta2_grad_current = del_J_del_a3 .* del_a3_del_w2;
	%layer 2
	del_J_del_a2 = (del_J_del_a3' * del_a3_del_a2(:,[2:size(del_a3_del_a2,2)]))';
	Theta1_grad_current = del_J_del_a2 .* del_a2_del_w1;

	Theta2_grad = Theta2_grad + Theta2_grad_current;
	Theta1_grad = Theta1_grad + Theta1_grad_current;
	% s1
	% s2
	% s3
	% disp("Theta 1"); size(Theta1)
	% disp("Theta 1 grad"); size(Theta1_grad)
	% disp("Theta 2"); size(Theta2)
	% disp("Theta 2 grad"); size(Theta2_grad)
	% pause
end;
J = (-1/m)*J_total + (lambda/(2*m))*(sum(sum(Theta1(:,[2:size(Theta1,2)]).^2,1),2) + sum(sum(Theta2(:,[2:size(Theta2,2)]).^2,1),2));
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,[2:size(Theta2,2)])];
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,[2:size(Theta1,2)])];













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
