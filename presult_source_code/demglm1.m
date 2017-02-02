%DEMGLM1 Demonstrate simple classification using a generalized linear model.
%
%	Description
%	 The problem consists of a two dimensional input matrix DATA and a
%	vector of classifications T.  The data is  generated from two
%	Gaussian clusters, and a generalized linear model with logistic
%	output is trained using iterative reweighted least squares. A plot of
%	the data together with the 0.1, 0.5 and 0.9 contour lines of the
%	conditional probability is generated.
%
%	See also
%	DEMGLM2, GLM, GLMTRAIN
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)


% Generate data from two classes in 2d
input_dim = 2;

% Fix seeds for reproducible results
randn('state', 42);
rand('state', 42);

ndata = 100;
% Generate mixture of two Gaussians in two dimensional space
data = randn(ndata, input_dim);
targets = zeros(ndata, 1);

% Priors for the two clusters
prior(1) = 0.4;
prior(2) = 0.6;

% Cluster centres
c = [2.0, 2.0; 0.0, 0.0];

ndata1 = prior(1)*ndata;
% Put first cluster at (2, 2)
data(1:ndata1, 1) = data(1:ndata1, 1) * 0.5 + c(1,1);
data(1:ndata1, 2) = data(1:ndata1, 2) * 0.5 + c(1,2);

% Leave second cluster at (0,0)
data((ndata1 + 1):(prior(2)+prior(1))*ndata, :) = ...
  data((ndata1 + 1):(prior(2)+prior(1))*ndata, :);
targets((ndata1+1):ndata, :) = 1;

% Plot the result

clc
disp('This demonstration illustrates the use of a generalized linear model')
disp('to classify data from two classes in a two-dimensional space. We')
disp('begin by generating and plotting the data.')
disp(' ')
disp('Press any key to continue.')
pause

fh1 = figure;
plot(data(1:ndata1,1), data(1:ndata1,2), 'bo');
hold on
axis([-4 5 -4 5])
set(gca, 'box', 'on')
plot(data(ndata1+1:ndata,1), data(ndata1+1:ndata,2), 'rx')
title('Data')

clc
disp('Now we fit a model consisting of a logistic sigmoid function of')
disp('a linear combination of the input variables.')
disp(' ')
disp('The model is trained using the IRLS algorithm for 5 iterations')
disp(' ')
disp('Press any key to continue.')
pause

net = glm(input_dim, 1, 'logistic');
options = foptions;
options(1) = 1;
options(14) = 5;
net = glmtrain(net, options, data, targets);

disp(' ')
disp('We now plot some density contours given by this model.')
disp('The contour labelled 0.5 is the decision boundary.')
disp(' ')
disp('Press any key to continue.')
pause
x = -4.0:0.2:5.0;
y = -4.0:0.2:5.0;
[X, Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);
grid = [X Y];
Z = glmfwd(net, grid);
Z = reshape(Z, length(x), length(y));
v = [0.1 0.5 0.9];
[c, h] = contour(x, y, Z, v);
title('Generalized Linear Model')
set(h, 'linewidth', 3)
clabel(c, h);

clc
disp('Note that the contours of constant density are straight lines.')
disp(' ')
disp('Press any key to end.')
pause
close(fh1);
clear all;

