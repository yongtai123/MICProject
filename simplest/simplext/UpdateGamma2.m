function gamma2 = UpdateGamma2(numAgents, gamma2, Adjacency, w, phi, niu)

    for k = 1:numAgents
        for l = 1:numAgents
            if Adjacency(l,k) == 1
                gamma2(l,k) = (1-niu)*gamma2(l,k) + niu*(norm(w(:,k)-phi(:,l))^2);
            end
        end
    end