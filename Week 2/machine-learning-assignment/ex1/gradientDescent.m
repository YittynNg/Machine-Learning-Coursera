function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    disp("Theta 1:")
    disp(theta(1))
    disp("Theta 2:") 
    disp(theta(2))
    
    disp("Diff 1:")
    diff1 = X*theta - y
    disp("Diff 2:")
    diff2 = ( X*theta - y ) .* X(:,2) 
    
    temp1 = theta(1) - alpha * (1/m) * sum(diff1)
    temp2 = theta(2) - alpha * (1/m) * sum(diff2)
    theta(1) = temp1
    theta(2) = temp2 
    



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
