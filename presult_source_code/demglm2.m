%DEMGLM2 Demonstrate simple classification using a generalized linear model.
%
%	Description
%	 The problem consists of a two dimensional input matrix DATA and a
%	vector of classifications T.  The data is  generated from three
%	Gaussian clusters, and a generalized linear model with softmax output
%	is trained using iterative reweighted least squares. A plot of the
%	data together with regions shaded by the classification given by the
%	network is generated.
%
%	See also
%	DEMGLM1, GLM, GLMTRAIN
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)


% Generate data from three classes in 2d
input_dim = 2;

% Fix seeds for reproducible results
randn('state', 42);
rand('state', 42);

ndata = 100;
% Generate mixture of three Gaussians in two dimensional space
data = randn(ndata, input_dim);
targets = zeros(ndata, 3);

% Priors for the clusters
prior(1) = 0.4;
prior(2) = 0.3;
prior(3) = 0.3;

% Cluster centres
c = [2.0, 2.0; 0.0, 0.0; 1, -1];

ndata1 = prior(1)*ndata;
ndata2 = (prior(1) + prior(2))*ndata;
% Put first cluster at (2, 2)
data(1:ndata1, 1) = data(1:ndata1, 1) * 0.5 + c(1,1);
data(1:ndata1, 2) = data(1:ndata1, 2) * 0.5 + c(1,2);
targets(1:ndata1, 1) = 1;

% Leave second cluster at (0,0)
data((ndata1 + 1):ndata2, :) = ...
  data((ndata1 + 1):ndata2, :);
targets((ndata1+1):ndata2, 2) = 1;

data((ndata2+1):ndata, 1) = data((ndata2+1):ndata,1) *0.6 + c(3, 1);
data((ndata2+1):ndata, 2) = data((ndata2+1):ndata,2) *0.6 + c(3, 2);
targets((ndata2+1):ndata, 3) = 1;

% Plot the result

clc
disp('This demonstration illustrates the use of a generalized linear model')
disp('to classify data from three classes in a two-dimensional space. We')
disp('begin by generating and plotting the data.')
disp(' ')
disp('Press any key to continue.')
pause

fh1 = figure;
plot(data(1:ndata1,1), data(1:ndata1,2), 'bo');
hold on
axis([-4 5 -4 5]);
set(gca, 'Box', 'on')
plot(data(ndata1+1:ndata2,1), data(ndata1+1:ndata2,2), 'rx')
plot(data(ndata2+1:ndata, 1), data(ndata2+1:ndata, 2), 'go')
title('Data')

clc
disp('Now we fit a model consisting of a softmax function of')
disp('a linear combination of the input variables.')
disp(' ')
disp('The model is trained using the IRLS algorithm for 5 iterations')
disp(' ')
disp('Press any key to continue.')
pause

net = glm(input_dim, size(targets, 2), 'softmax');
options = foptions;
options(1) = 1;
options(14) = 5;
net = glmtrain(net, options, data, targets);

disp(' ')
disp('We now plot the decision regions given by this model.')
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
[foo , class] = max(Z');
class = class';
colors = ['b.'; 'r.'; 'g.'];
for i = 1:3
  thisX = X(class == i);
  thisY = Y(class == i);
  h = plot(thisX, thisY, colors(i,:));
  set(h, 'MarkerSize', 8);
end
title('Plot of Decision regions')

hold off

clc
disp('Note that the boundaries of decision regions are straight lines.')
disp(' ')
disp('Press any key to end.')
pause
close(fh1);
clear all; 

