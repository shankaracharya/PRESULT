function a = gmmactiv(mix, x)
%GMMACTIV Computes the activations of a Gaussian mixture model.
%
%	Description
%	This function computes the activations A (i.e. the  probability
%	P(X|J) of the data conditioned on each component density)  for a
%	Gaussian mixture model.   The data structure MIX defines the mixture
%	model, while the matrix X contains the data vectors.  Each row of X
%	represents a single vector.
%
%	See also
%	GMM, GMMPOST, GMMPROB
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

ndata = size(x, 1);

switch mix.covar_type

  case 'spherical'
    % Calculate squared norm matrix, of dimension (ndata, ncentres)
    n2 = dist2(x, mix.centres);

    % Calculate width factors
    wi2 = ones(ndata, 1) * (2 .* mix.covars);
    normal = (pi .* wi2) .^ (mix.nin/2);

    % Now compute the activations
    a = exp(-(n2./wi2))./ normal;

  case 'diag'
    a = zeros(ndata, mix.ncentres);
    normal = (2*pi)^(mix.nin/2);
    s = prod(sqrt(mix.covars), 2);
    for i = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(i, :));
      a(:, i) = exp(-0.5*sum((diffs.*diffs)./(ones(ndata, 1) * ...
	mix.covars(i,:)), 2)) ./ (normal*s(i));
    end
    
  case 'full'
    a = zeros(ndata, mix.ncentres);
    normal = (2*pi)^(mix.nin/2);
    for i = 1:mix.ncentres
      diffs = x - (ones(ndata, 1) * mix.centres(i,:));
      % Use Cholesky decomposition of covariance matrix to speed computation
      c = chol(mix.covars(:,:,i));
      temp = diffs/c;
      a(:,i) = exp(-0.5*sum(temp.*temp, 2))./(normal*prod(diag(c)));
    end
end
  
