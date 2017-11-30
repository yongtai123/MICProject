function w = UpdateOmega(numAgents, A, w,  phi)

    for k = 1:numAgents
        w(:,k) = phi*A(:,k);
    end