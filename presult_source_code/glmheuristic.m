function net = glmheuristic(net, x, t)
%glmheuristic Specialised training of a GLM gating network with one-pass least squares for the M-step of a ME trained by EM
%
%	NET = GLMHEURISTIC(NET, X, T) uses a one-pass least squares
%	algorithm to set the weights in the generalized linear model
%	structure NET.  This is a more efficient alternative to using MIXERR
%	and MIXGRAD and a non-linear optimisation routine through
%	MIXOPT. Each row of X corresponds to one input vector and each row
%	of T corresponds to one target vector.
%
%	See also
%	glmtrain, mixem, mix 
%
%       Copyright: Perry Moerland

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~errstring
  error(errstring);
end

ndata = size(x, 1);
% Add a column of ones for the bias 
inputs = [x ones(ndata, 1)];

% one-step solution
% ensure that log(t) is computable
t(t<realmin) = realmin;

temp = inputs\log(t);
net.w1 = temp(1:net.nin, :);
net.b1 = temp(net.nin+1, :);


