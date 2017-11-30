function gamma2 = UpdateGamma2(attacker, attacker_phi, numAgents, gamma2, Adjacency, w, phi, niu)

    for k = 1:numAgents
        if ~any(k == attacker)
            for l = 1:numAgents
                if Adjacency(l,k) == 1
                    if any(attacker == l)
                        gamma2(l,k) = (1-niu)*gamma2(l,k) + niu*(norm(w(:,k)-attacker_phi(:,k))^2);
                    else
                        gamma2(l,k) = (1-niu)*gamma2(l,k) + niu*(norm(w(:,k)-phi(:,l))^2);
                    end
                end
            end
        end
    end