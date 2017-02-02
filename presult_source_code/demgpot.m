function g = demgpot(x, mix)
%DEMGPOT Computes the gradient of the negative log likelihood for a mixture model.
%
%	Description
%	This function computes the gradient of the negative log of the
%	unconditional data density P(X) for a Gaussian mixture model.  The
%	data structure MIX defines the mixture model, while the matrix X
%	contains the data vectors.
%
%	See also
%	DEMHMC1, DEMMET1, DEMPOT
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Computes the potential gradient

temp = (ones(mix.ncentres,1)*x)-mix.centres;
temp = temp.*(gmmactiv(mix,x)'*ones(1, mix.ncentres));
% Assume spherical covariance structure
if ~strcmp(mix.covar_type, 'spherical')
  error('Spherical covariance only.')
end
temp = temp./(mix.covars'*ones(1, mix.ncentres));
temp = temp.*(mix.priors'*ones(1, mix.ncentres));
g = sum(temp, 1)/gmmprob(mix, x);