function [net, gamma] = evidence(net, x, t, num)
%EVIDENCE Re-estimate hyperparameters using evidence approximation.
%
%	Description
%	[NET, GAMMA] = EVIDENCE(NET, X, T) re-estimates the hyperparameters
%	ALPHA and BETA by applying Bayesian re-estimation formulae for NUM
%	iterations. The hyperparameter ALPHA can be a simple scalar
%	associated with an isotropic prior on the weights, or can be a vector
%	in which each component is associated with a group of weights as
%	defined by the INDEX matrix in the NET data structure. These more
%	complex priors can be set up for an MLP using MLPPRIOR. Initial
%	values for the iterative re-estimation are taken from the network
%	data structure NET passed as an input argument, while the return
%	argument NET contains the re-estimated values.
%
%	[NET, GAMMA] = EVIDENCE(NET, X, T, NUM) allows the re-estimation
%	formula to be applied for NUM cycles in which the re-estimated values
%	for the hyperparameters from each cycle are used to re-evaluate the
%	Hessian matrix for the next cycle.
%
%	See also
%	MLPPRIOR, NETGRAD, NETHESS, DEMEV1, DEMARD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

errstring = consist(net, '', x, t);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);
if nargin == 3
  num = 1;
end

% Extract weights from network
pakstr = [net.type, 'pak'];
w = feval(pakstr, net);

% Evaluate data-dependent contribution to the Hessian matrix.
[h, dh] = nethess(w, net, x, t); 

% Now set the negative eigenvalues to zero.
[evec, eval] = eig(dh);
eval = eval.*(eval > 0);

% Do the re-estimation. 
for k = 1 : num
  [e, edata, eprior] = neterr(w, net, x, t);
  h = nethess(w, net, x, t, dh);
  % Re-estimate alpha.
  if size(net.alpha) == [1 1]
    % Evaluate number of well-determined parameters.
    if k == 1
      % Form vector of eigenvalues
      eval = diag(eval);
    end
    B = net.beta*eval;
    gamma = sum(B./(B + net.alpha));
    net.alpha = 0.5*gamma/eprior;
  else
    ngroups = size(net.alpha, 1);
    gams = zeros(1, ngroups);
    % Reconstruct data hessian with negative eigenvalues set to zero.
    dh = evec*eval*evec';
    h = nethess(w, net, x, t, dh);
    for m = 1 : ngroups
      gams(m) = sum(net.index(:,m)) - ...
	        net.alpha(m)*sum(diag(inv(h)).*net.index(:,m));
      net.alpha(m) = gams(m)/(2*eprior(m));
    end 
    gamma = sum(gams, 2);
  end
  % Re-estimate beta.
  net.beta = 0.5*(net.nout*ndata - gamma)/edata;
end

