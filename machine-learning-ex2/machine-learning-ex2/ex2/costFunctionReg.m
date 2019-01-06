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

theta_aug = theta(2:length(theta));
X_aug = X(:,2:size(X, 2);
[J, grad] = costFunction(theta_aug, X_aug, y);

J = J + (lambda/(2*m)) * sum(theta_aug .^ 2);

for weight = 1:length(theta_aug)
    grad(weight) = (grad(weight) + ((lambda/m) * theta_aug(weight)));
end



% =============================================================

end
