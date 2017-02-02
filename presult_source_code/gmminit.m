function mix = gmminit(mix, x, options)
%GMMINIT Initialises Gaussian mixture model from data
%
%	Description
%	MIX = GMMINIT(MIX, X, OPTIONS) uses a dataset X to initialise the
%	parameters of a Gaussian mixture model defined by the data structure
%	MIX.  The k-means algorithm is used to determine the centres. The
%	priors are computed from the proportion of examples belonging to each
%	cluster. The covariance matrices are calculated as the sample
%	covariance of the points associated with (i.e. closest to) the
%	corresponding centres. This initialisation can be used as the
%	starting point for training the model using the EM algorithm.
%
%	See also
%	GMM
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

[ndata, xdim] = size(x);

% Check that inputs are consistent
errstring = consist(mix, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

% Arbitrary width used if variance collapses to zero: make it 'large' so
% that centre is responsible for a reasonable number of points.
WIDTH = 1.0;

perm = randperm(ndata);
perm = perm(1:mix.ncentres);

% Assign first ncentres (permuted) data points as centres
mix.centres = x(perm, :);
options(5) = 0;	

% Use kmeans algorithm to set centres
[mix.centres, options, post] = kmeans(mix.centres, x, options);

% Set priors depending on number of points in each cluster
cluster_sizes = max(sum(post, 1), 1);  % Make sure that no prior is zero
mix.priors = cluster_sizes/sum(cluster_sizes); % Normalise priors

switch mix.covar_type
  case 'spherical'
    % Determine widths as distance to nearest centre 
    % (or a constant if this is zero)
    for i = 1:mix.ncentres
      cdist = zeros(1, mix.ncentres);
      % First calculate minimum distance from each centre to the rest
      for j = 1:mix.ncentres
        if i ~= j
          cdist(j) = sum((mix.centres(i, :) - mix.centres(j, :)).^ 2);
        end
      end

      % Make sure that the distance of c(i) to itself does not get used
      cdist(i) = [];
      mix.covars(i) = sqrt(min(cdist));
      % If width is small, arbitrarily re-assign to width WIDTH
      if mix.covars(i) < eps
        mix.covars(i) = WIDTH;
      end
    end
  case 'diag'
    for i = 1:mix.ncentres
      % Pick out data points belonging to this centre
      c = x(find(post(:,i)),:);
      diffs = x - (ones(size(x, 1), 1) * mix.centres(i, :));
      mix.covars(i,:) = sum((diffs.*diffs), 1)/size(x, 1);
      % Replace small entries by WIDTH value
      mix.covars(i,:) = mix.covars(i,:) + WIDTH.*(mix.covars(i,:)<eps);
    end
  case 'full'
    for i = 1:mix.ncentres
      % Pick out data points belonging to this centre
      c = x(find(post(:,i)),:);
      diffs = x - (ones(size(x, 1), 1) * mix.centres(i, :));
      mix.covars(:,:,i) = (diffs'*diffs)/(size(x, 1));
      % Add WIDTH*Identity to rank-deficient covariance matrices
      if rank(mix.covars(:,:,i)) < mix.nin
	mix.covars(:,:,i) = mix.covars(:,:,i) + WIDTH.*eye(mix.nin);
      end
    end
end

