function gradchek(w, func, grad, varargin)
%GRADCHEK Checks a user-defined gradient function using finite differences.
%
%	Description
%	This function is intended as a utility for other netlab functions
%	(particularly optimisation functions) to use.  It enables the user to
%	check whether a gradient calculation has been correctly implmented
%	for a given function. GRADCHEK(W, FUNC, GRAD) checks how accurate the
%	gradient  GRAD of a function FUNC is at a parameter vector X.   A
%	central difference formula with step size 1.0e-6 is used, and the
%	results for both gradient function and finite difference
%	approximation are printed. There is no return value.
%
%	GRADCHEK(X, FUNC, GRAD, P1, P2, ...) allows additional arguments to
%	be passed to FUNC and GRAD.
%
%	See also
%	CONJGRAD, GRADDESC, HMC, OLGD, QUASINEW, SCG
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

% Reasonable value for step size
epsilon = 1.0e-6;

func = fcnchk(func, length(varargin));
grad = fcnchk(grad, length(varargin));

% Treat
nparams = length(w);
deltaf = zeros(1, nparams);
step = zeros(1, nparams);
for i = 1:nparams
  % Move a small way in the ith coordinate of w
  step(i) = 1.0;
  fplus  = feval('linef', epsilon, func, w, step, varargin{:});
  fminus = feval('linef', -epsilon, func, w, step, varargin{:});
  % Use central difference formula for approximation
  deltaf(i) = 0.5*(fplus - fminus)/epsilon;
  step(i) = 0.0;
end
gradient = feval(grad, w, varargin{:});
fprintf(1, 'Checking gradient ...\n\n');
fprintf(1, '   analytic   diffs     delta\n\n');
disp([gradient', deltaf', gradient' - deltaf'])