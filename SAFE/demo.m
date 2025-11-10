clear;clc;

%load dataset (lost)
load('lost sample.mat');

% hyper-parameters  in readme
Maxiter = 15;
k = 10;
alpha = 30;
beta = 0.5;
gamma = 0.001;
lambda = 0.03;


train_p_target = train_p_target';
test_target = test_target';
train_target=train_target';


[ accuracy_test] = SAFE(train_data, train_p_target, ...
    test_data, test_target,train_target, k,  Maxiter, ...
    gamma, lambda, alpha, beta);

