%
%	Netlab Functions
%
%	An alphabetic list of functions in Netlab.
% conffig  -  Display a confusion matrix. 
% confmat  -  Compute a confusion matrix. 
% conjgrad -  Conjugate gradients optimization. 
% consist  -  Check that arguments are consistent. 
% datread  -  Read data from an ascii file. 
% datwrite -  Write data to ascii file. 
% dem2ddat -  Generates two dimensional data for demos. 
% demard   -  Automatic relevance determination using the MLP. 
% demev1   -  Demonstrate Bayesian regression for the MLP. 
% demgauss -  Demonstrate sampling from Gaussian distributions. 
% demglm1  -  Demonstrate simple classification using a generalized linear model. 
% demglm2  -  Demonstrate simple classification using a generalized linear model. 
% demgmm1  -  Demonstrate density modelling with a Gaussian mixture model. 
% demgmm2  -  Demonstrate density modelling with a Gaussian mixture model. 
% demgmm3  -  Demonstrate density modelling with a Gaussian mixture model. 
% demgpot  -  Computes the gradient of the negative log likelihood for a mixture model. 
% demhint  -  Demonstration of Hinton diagram for 2-layer feed-forward network. 
% demhmc1  -  Demonstrate Hybrid Monte Carlo sampling on mixture of two Gaussians. 
% demhmc2  -  Demonstrate Bayesian regression with Hybrid Monte Carlo sampling. 
% demhmc3  -  Demonstrate Bayesian regression with Hybrid Monte Carlo sampling. 
% demkmean -  Demonstrate simple clustering model trained with K-means. 
% demknn1  -  Demonstrate nearest neighbour classifier. 
% demmdn1  -  Demonstrate fitting a multi-valued function using a Mixture Density Network. 
% demmet1  -  Demonstrate Markov Chain Monte Carlo sampling on a Gaussian. 
% demmlp1  -  Demonstrate simple regression using a multi-layer perceptron 
% demmlp2  -  Demonstrate simple classification using a multi-layer perceptron 
% demnlab  -  A front-end Graphical User Interface to the demos 
% demolgd1 -  Demonstrate simple MLP optimisation with on-line gradient descent 
% demopt1  -  Demonstrate different optimisers on Rosenbrock's function. 
% dempot   -  Computes the negative log likelihood for a mixture model. 
% demprior -  Demonstrate sampling from a multi-parameter Gaussian prior. 
% demrbf1  -  Demonstrate simple regression using a radial basis function network. 
% demtrain -  Demonstrate training of MLP network. 
% dist2    -  Calculates squared distance between two sets of points. 
% evidence -  Re-estimate hyperparameters using evidence approximation. 
% gauss    -  Evaluate a Gaussian distribution. 
% glm      -  Create a generalized linear model. 
% glmerr   -  Evaluate error function for generalized linear model. 
% glmfwd   -  Forward propagation through generalized linear model. 
% glmgrad  -  Evaluate gradient of error function for generalized linear model. 
% glmhess  -  Evaluate the Hessian matrix for a generalised linear model. 
% glminit  -  Initialise the weights in a generalized linear model. 
% glmpak   -  Combines weights and biases into one weights vector. 
% glmtrain -  Specialised training of generalized linear model 
% glmunpak -  Separates weights vector into weight and bias matrices. 
% gmm      -  Creates a Gaussian mixture model with specified architecture. 
% gmmactiv -  Computes the activations of a Gaussian mixture model. 
% gmmem    -  EM algorithm for Gaussian mixture model. 
% gmminit  -  Initialises Gaussian mixture model from data 
% gmmpak   -  Combines all the parameters in a Gaussian mixture model into one vector. 
% gmmpost  -  Computes the class posterior probabilities of a Gaussian mixture model. 
% gmmprob  -  Computes the data probability for a Gaussian mixture model. 
% gmmsamp  -  Sample from a Gaussian mixture distribution. 
% gmmunpak -  Separates a vector of Gaussian mixture model parameters into its components. 
% gradchek -  Checks a user-defined gradient function using finite differences. 
% graddesc -  Gradient descent optimization. 
% gsamp    -  Sample from a Gaussian distribution. 
% hesschek -  Use central differences to confirm correct evaluation of Hessian matrix. 
% hintmat  -  Evaluates the coordinates of the patches for a Hinton diagram. 
% hinton   -  Plot Hinton diagram for a weight matrix. 
% histp    -  Histogram estimate of 1-dimensional probability distribution. 
% hmc      -  Hybrid Monte Carlo sampling. 
% kmeans   -  Trains a k means cluster model. 
% knn      -  Creates a K-nearest-neighbour classifier. 
% linef    -  Calculate function value along a line. 
% linemin  -  One dimensional minimization. 
% mdn      -  Creates a Mixture Density Network with specified architecture. 
% mdnerr   -  Evaluate error function for Mixture Density Network. 
% mdnfwd   -  Forward propagation through Mixture Density Network. 
% mdngrad  -  Evaluate gradient of error function for Mixture Density Network. 
% mdninit  -  Initialise the weights in a Mixture Density Network. 
% mdnpak   -  Combines weights and biases into one weights vector. 
% mdnunpak -  Separates weights vector into weight and bias matrices. 
% metrop   -  Markov Chain Monte Carlo sampling with Metropolis algorithm. 
% minbrack -  Bracket a minimum of a function of one variable. 
% mlp      -  Create a 2-layer feedforward network. 
% mlpbkp   -  Backpropagate gradient of error function for 2-layer network. 
% mlpderiv -  Evaluate derivatives of network outputs with respect to weights. 
% mlperr   -  Evaluate error function for 2-layer network. 
% mlpfwd   -  Forward propagation through 2-layer network. 
% mlpgrad  -  Evaluate gradient of error function for 2-layer network. 
% mlphdotv -  Evaluate the product of the data Hessian with a vector. 
% mlphess  -  Evaluate the Hessian matrix for a multi-layer perceptron network. 
% mlphint  -  Plot Hinton diagram for 2-layer feed-forward network. 
% mlpinit  -  Initialise the weights in a 2-layer feedforward network. 
% mlppak   -  Combines weights and biases into one weights vector. 
% mlpprior -  Create Gaussian prior for mlp. 
% mlptrain -  Utility to train an MLP network for demtrain 
% mlpunpak -  Separates weights vector into weight and bias matrices. 
% neterr   -  Evaluate network error function for generic optimizers 
% netgrad  -  Evaluate network error gradient for generic optimizers 
% nethess  -  Evaluate network Hessian 
% netopt   -  Optimize the weights in a network model. 
% olgd     -  On-line gradient descent optimization. 
% plotmat  -  Display a matrix. 
% quasinew -  Quasi-Newton optimization. 
% rbf      -  Creates an RBF network with specified architecture 
% rbferr   -  Evaluate error function for RBF network. 
% rbffwd   -  Forward propagation through RBF network with linear outputs. 
% rbfgrad  -  Evaluate gradient of error function for RBF network. 
% rbfpak   -  Combines all the parameters in an RBF network into one weights vector. 
% rbftrain -  Two stage training of RBF network. 
% rbfunpak -  Separates a vector of RBF weights into its components. 
% rosegrad -  Calculate gradient of Rosenbrock's function. 
% rosen    -  Calculate Rosenbrock's function. 
% scg      -  Scaled conjugate gradient optimization. 
%
%	Copyright (c) Christopher M Bishop Ian T Nabney (1996, 1997)
%
