function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

multiple = (1 + exp(-z));
g = 1./multiple;
return



end
