function mix = gmm(dim, ncentres, covar_type)
%GMM	Creates a Gaussian mixture model with specified architecture.
%
%	Description
%	 MIX = GMM(DIM, NCENTRES, COVAR_TYPE) takes the dimension of the
%	space DIM, the number of centres in the mixture model and the type of
%	the mixture model, and returns a data structure MIX. The mixture
%	model type defines the covariance structure of each component
%	Gaussian:
%	  'spherical' = single variance parameter for each component: stored as a vector
%	  'diag' = diagonal matrix for each component: stored as rows of a matrix
%	  'full' = full matrix for each component: stored as 3d array
%
%	The priors are initialised to equal values summing to one, and the
%	covariances are all the identity matrix (or equivalent).  The centres
%	are initialised randomly from a zero mean unit variance Gaussian.
%	This makes use of the MATLAB function RANDN and so the seed for the
%	random weight initialisation can be set using RANDN('STATE', S) where
%	S is the state value.
%
%	The fields in MIX are
%	  
%	  type = 'gmm'
%	  nin = the dimension of the space
%	  ncentres = number of mixture components
%	  covar_type = string for type of variance model
%	  priors = mixing coefficients
%	  centres = means of Gaussians: stored as rows of a matrix
%	  covars = covariances of Gaussians
%
%	See also
%	GMMPAK, GMMUNPAK, GMMSAMP, GMMINIT, GMMEM, GMACTIV, GMPOST, 
%	GMPROB
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

mix.type = 'gmm';
mix.nin = dim;
mix.ncentres = ncentres;
mix.nparams = mix.ncentres + mix.ncentres*mix.nin + mix.ncentres;

vartypes = {'spherical', 'diag', 'full'};

if sum(strcmp(covar_type, vartypes)) == 0
  error('Undefined covariance type')
else
  mix.covar_type = covar_type;
end

% Initialise priors to be equal and summing to one
mix.priors = ones(1,mix.ncentres) ./ mix.ncentres;

% Initialise centres
mix.centres = randn(mix.ncentres, mix.nin);

% Initialise all the variances to unity
switch mix.covar_type

  case 'spherical'
    mix.covars = ones(1, mix.ncentres);
  case 'diag'
    % Store diagonals of covariance matrices as rows in a matrix
    mix.covars =  ones(mix.ncentres, mix.nin);
  case 'full'
    % Store covariance matrices in a row vector of matrices
    mix.covars = repmat(eye(mix.nin), [1 1 mix.ncentres]);
    
end

