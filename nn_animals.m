% Initialization
clear ; close all; clc

load('data.mat');
%% build a network
%net=feedforwardnet(25);
net=patternnet([25,10,10]);

% set the learning rate
net.trainParam.lr=0.005;


% divided into training, validation and testing simulate
rand_indices=randperm(3000);

trainData=X(:,rand_indices(1:2400));
trainLabels=y(:,rand_indices(1:2400));
testData = X(:, rand_indices(2401:end));
testLabels = y(:, rand_indices(2401:end));

% train a neural network
net = train(net,trainData,trainLabels);

%% show the network
view(net);

preds = net(testData);

% confusion matrix
figure(2);
plotconfusion(testLabels, preds);