clear; close all; clc
rng('default');

%% NETWORK TOPOLOGY FROM WEBGME 
Adjacency = getAdjacency();


%% PARAMETERS
numAgents = size(Adjacency,1);
numPoints = 5000;
numTaps = 2;		% channel number
Mu = 0.01;          % step size
niu = 0.02;         % forgetting factor
w = rand(numTaps,numAgents);    
w0 = rand(numTaps, 1);          % goal state
phi = zeros(numTaps,numAgents);
gamma2 = zeros(numAgents,numAgents);


%% INPUTS (GAUSSIAN)
mu_x = 0;
sigma_x2 = 0.8 + 0.4*rand(numAgents,1);
x = zeros(numPoints,numAgents);
for k = 1:numAgents
    x(:,k) = mvnrnd(mu_x, sigma_x2(k), numPoints);
end
v = 0.04*randn(numPoints,numAgents);
d = zeros(numPoints,numAgents);
for k = 1:numAgents
    d(:,k) = filter(w0, 1, x(:,k));
end
d = d+v;


%% DIFFUSION LMS ALGORITHM
for n = numTaps : numPoints
    for k = 1 : numAgents
        u(:,k) = x(n:-1:n-numTaps+1,k);     % select part of training input
        h(n,k) = u(:,k)'*w(:,k);            % hypothsis function
        e(n,k) = d(n,k)-h(n,k);             % error
        phi(:,k) = w(:,k) + Mu*( u(:,k)*e(n,k) );
    end
    
    gamma2 = UpdateGamma2(numAgents, gamma2, Adjacency, w, phi, niu);
    A = UpdateWeight(numAgents, gamma2, Adjacency);
    w = UpdateOmega(numAgents, A, w, phi);
end


%% PLOT
figure(1);
plot(abs(e));
ylabel('Output Estimation Error');
title(['LMS Adaptation Learning Curve Using Mu = ', num2str(Mu)]);
xlabel('Iteration Number');
saveas(gcf,'network.jpg');

