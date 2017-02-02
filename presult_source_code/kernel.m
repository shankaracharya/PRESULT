function mix = kernel(mix, x, post, options)
%kernel	One-step update of the parameters of a Gaussian kernel gating network in a mixture of experts model trained by EM
%
%	MIX = KERNEL(MIX, X, POST, OPTIONS) updates the parameters (mixing
%       coefficients, centres, covariances) of a Gaussian kernel gating
%       network MIX in a mixture of experts model trained by EM. The
%	matrix X represents the data whose expectation is maximized, with
%	each row corresponding to a vector and the matrix POST gives the
%       posterior probabilities (result of the E-step). The optional
%       parameters have the following interpretations:
%
%	OPTIONS(5) is set to 1 if a covariance matrix is reset to its
%	original value when any of its singular values are too small (less
%	than eps).  With the default value of 0 no action is taken.
%
%       This is a stripped down version of Netlab's 'gmmem' EM algorithm for
%       a Gaussian mixture model.
%
%	See also
%       mixem, mix, gmmem, gmm
%
%       Copyright: Perry Moerland

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

[ndata, xdim] = size(x);

display = options(1);
options(5) = 1;

check_covars = 0;
if options(5) >= 1
  %disp('check_covars is on');
  check_covars = 1;	% Ensure that covariances don't collapse
  MIN_COVAR = eps;	% Minimum singular value of covariance matrix
  init_covars = mix.covars;
end

  % Adjust the new estimates for the parameters
  new_pr = sum(post, 1);
  new_c = post' * x;
    
  % Now move new estimates to old parameter vectors
  mix.priors = new_pr ./ ndata;
  mix.centres = new_c ./ (new_pr' * ones(1, mix.nin));
 
  switch mix.covar_type
    case 'spherical'
      n2 = dist2(x, mix.centres);
      for j = 1:mix.ncentres
        v(j) = (post(:,j)'*n2(:,j));
      end
      mix.covars = ((v./new_pr))./mix.nin;
      if check_covars
	% Ensure that no covariance is too small
	for j = 1:mix.ncentres
	  if mix.covars(j) < MIN_COVAR
	    mix.covars(j) = init_covars(j);
	  end
	end
      end
    case 'diag'
      for j = 1:mix.ncentres
	diffs = x - (ones(ndata, 1) * mix.centres(j,:));
	mix.covars(j,:) = sum((diffs.*diffs).*(post(:,j)*ones(1, ...
	  mix.nin)), 1)./new_pr(j);
      end
      if check_covars
	% Ensure that no covariance is too small
	for j = 1:mix.ncentres
	  if min(mix.covars(j,:)) < MIN_COVAR
	    mix.covars(j,:) = init_covars(j,:);
	  end
	end
      end
    case 'full'
      for j = 1:mix.ncentres
        diffs = x - (ones(ndata, 1) * mix.centres(j,:));
        diffs = diffs.*(sqrt(post(:,j))*ones(1, mix.nin));
        mix.covars(:,:,j) = (diffs'*diffs)/new_pr(j);
      end
      if check_covars
	% Ensure that no covariance is too small
	for j = 1:mix.ncentres
	  if min(svd(mix.covars(:,:,j))) < MIN_COVAR
	    mix.covars(:,:,j) = init_covars(:,:,j);
	  end
	end
      end
  end
