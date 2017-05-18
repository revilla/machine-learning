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

hX = sigmoid(X*theta);
nJ = (-y'*log(hX)-(1-y)'*log(1-hX))/m;
thetaprime = theta([2:end]);
Xprime = X([1:end 2:end]);
hXprime = sigmoid(Xprime*thetaprime);
reg = (lambda/(2*m))*(thetaprime'*thetaprime);

J = nJ - reg;

grad(1) = ((1/m).*(hX - y)'*X)(1);

grad([2:end 1]) = ((1/m).*(hXprime - y)'*Xprime)([2:end 1]) + (lambda/m)*thetaprime;

% =============================================================

end
