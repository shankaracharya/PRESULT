function [e, y, a, edata, eprior] = glmerr(net, x, t)
%GLMERR	Evaluate error function for generalized linear model.
%
%	Description
%	 E = GLMERR(NET, X, T) takes a generalized linear model data
%	structure NET together with a matrix X of input vectors and a matrix
%	T of target vectors, and evaluates the error function E. The choice
%	of error function corresponds to the output unit activation function.
%	Each row of X corresponds to one input vector and each row of T
%	corresponds to one target vector.
%
%	[E, Y, A] = GLMERR(NET, X) also returns a matrix Y giving the outputs
%	of the models and a matrix A  giving the summed inputs to each output
%	unit, where each row corresponds to one pattern.
%
%	[E, Y, A, EDATA, EPRIOR] = GLMERR(NET, X, T) also returns the data
%	and prior components of the total error.
%
%	See also
%	GLM, GLMPAK, GLMUNPAK, GLMFWD, GLMGRAD, GLMTRAIN
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check arguments for consistency
errstring = consist(net, 'glm', x, t);
if ~isempty(errstring);
  error(errstring);
end

[y, a] = glmfwd(net, x);

switch net.actfn

  case 'linear'  	% Linear outputs

    e = 0.5*sum(sum((y - t).^2));

  case 'logistic'  	% Logistic outputs

    % Ensure that log(1-y) is computable: need exp(a) > eps
    maxcut = -log(eps);
    % Ensure that log(y) is computable
    mincut = -log(1/realmin - 1);
    a = min(a, maxcut);
    a = max(a, mincut);
    y = 1./(1 + exp(-a));
    e = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'   	% Softmax outputs
  
    nout = size(a,2);
    % Ensure that sum(exp(a), 2) does not overflow
    maxcut = log(realmax) - log(nout);
    % Ensure that exp(a) > 0
    mincut = log(realmin);
    a = min(a, maxcut);
    a = max(a, mincut);
    temp = exp(a);
    y = temp./(sum(temp, 2)*ones(1,nout));
    % Ensure that log(y) is computable
    y(y<realmin) = realmin;
    e = - sum(sum(t.*log(y)));

end

if isfield(net, 'beta')
  e1 = net.beta*e;
else
  e1 = e;
end

if isfield(net, 'alpha')
  w = glmpak(net);
  if size(net.alpha) == [1 1]
    eprior = 0.5*(w*w');
    e2 = eprior*net.alpha;
  else
    eprior = 0.5*(w.^2)*net.index;
    e2 = eprior*net.alpha;
  end
else
  eprior = 0;
  e2 = 0;
end

e = e1 + e2;

