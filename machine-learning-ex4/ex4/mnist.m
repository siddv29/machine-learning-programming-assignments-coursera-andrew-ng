X = loadImages('train-images-idx3-ubyte')';
y = loadLabels('train-labels-idx1-ubyte');
size(X)
size(y)
m = size(X, 1)
input_layer_size = 784;
hidden_layer_size = 20;
output_layer_size = 10;
theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
theta2 = randInitializeWeights(hidden_layer_size, num_labels);
theta_params = [theta1(:) ; theta2(:)];
options = optimset('MaxIter', 50);
lambda = 3;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, output_layer_size, X([1:10000],:), y, lambda);
[nn_params, cost] = fmincg(costFunction, theta_params, options);

theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
pred = predict(theta1,theta2,X([10001:11000],:));
actual = y([10001:11000],:)+1;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == actual)) * 100);
