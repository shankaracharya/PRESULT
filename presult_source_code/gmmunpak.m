function mix = gmmunpak(mix, p)
%GMMUNPAK Separates a vector of Gaussian mixture model parameters into its components.
%
%	Description
%	MIX = GMMUNPAK(MIX, P) takes a GMM data structure MIX and  a single
%	row vector of parameters P and returns a mixture data structure
%	identical to the input MIX, except that the mixing coefficients
%	PRIORS, centres CENTRES and covariances COVARS are all set to the
%	corresponding elements of P.
%
%	See also
%	GMM, GMMPAK
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

errstring = consist(mix, 'gmm');
if ~errstring
  error(errstring);
end

mark1 = mix.ncentres;
mark2 = mark1 + mix.ncentres*mix.nin;

mix.priors = reshape(p(1:mark1), 1, mix.ncentres);
mix.centres = reshape(p(mark1 + 1:mark2), mix.ncentres, mix.nin);
switch mix.covar_type
  case 'spherical'
    mark3 = mix.ncentres*(2 + mix.nin);
    mix.covars = reshape(p(mark2 + 1:mark3), 1, mix.ncentres);
  case 'diag'
    mark3 = mix.ncentres*(1 + mix.nin + mix.nin);
    mix.covars = reshape(p(mark2 + 1:mark3), mix.ncentres, mix.nin);
  case 'full'
    mark3 = mix.ncentres*(1 + mix.nin + mix.nin*mix.nin);
    mix.covars = reshape(p(mark2 + 1:mark3), mix.nin, mix.nin, ...
      mix.ncentres);
end
  
