clear; close all; clc
rng('default');

%% NETWORK TOPOLOGY FROM WEBGME 
[Adjacency, attacker, attackType] = getData();
numAgents = size(Adjacency,1);


%% PARAMETERS
numPoints = 5000;
numTaps = 2;		% channel number
Mu = 0.01;          % step size
niu = 0.02;         % forgetting factor
w = rand(numTaps,numAgents);    
w0 = rand(numTaps, 1);          % goal state
phi = zeros(numTaps,numAgents);
gamma2 = zeros(numAgents,numAgents);
AgentSet = 1:numAgents;
normalAgentSet = setdiff(AgentSet,attacker);


%% INPUTS (GAUSSIAN)
mu_x = 0;
sigma_x2 = 0.8 + 0.4*rand(numAgents,1);
x = zeros(numPoints,numAgents);
for k = 1:numAgents
    x(:,k) = mvnrnd(mu_x, sigma_x2(k), numPoints);
end
d = zeros(numPoints,numAgents);
for k = 1:numAgents
    d(:,k) = filter(w0, 1, x(:,k));
end

%% ATTACKER INPUTS
ra = 0.002;
attacker_phi = zeros(numTaps, numAgents);
w0_attacker = rand(numTaps, 1);          % attacker's goal state
for i = 1:size(attacker,2)
    if strcmp(attackType(i),'SimpleStaticObjectiveAttacker')
        d(:,attacker(i)) = filter(w0_attacker, 1, x(:,k));
        attacker(i) = [];
    end

end
v = 0.04*randn(numPoints,numAgents);        
d = d+v;

%% DIFFUSION LMS ALGORITHM
for n = numTaps : numPoints
    for k = 1 : numAgents
        u(:,k) = x(n:-1:n-numTaps+1,k);     % select part of training input
        h(n,k) = u(:,k)'*w(:,k);            % hypothsis function
        e(n,k) = d(n,k)-h(n,k);             % error
        phi(:,k) = w(:,k) + Mu*( u(:,k)*e(n,k) );
    end
    
    if size(attacker,2) ~= 0
        if strcmp(attackType,'ByzantineChangeObjectiveAttacker')
            for k = 1:numAgents 
                if ~any(attacker == k)
                    attacker_phi(:,k) = w(:,k) + ra*(w0_attacker-w(:,k));
                end
            end
        end
    end
            
    gamma2 = UpdateGamma2(attacker, attacker_phi, numAgents, gamma2, Adjacency, w, phi, niu);
    A = UpdateWeight(attacker, numAgents, gamma2, Adjacency);
    w = UpdateOmega(numAgents, A, w, phi);
end

%update adjacency matrix
Adjacency = (A>=0.05) & Adjacency;


%% PLOT
figure(1);
for k = normalAgentSet
    plot(abs(e(:,k)));
    hold on;
end
ylabel('Output Estimation Error');
title(['LMS Adaptation Learning Curve Using Mu = ', num2str(Mu)]);
xlabel('Iteration Number');
saveas(gcf,'Network.jpg');
