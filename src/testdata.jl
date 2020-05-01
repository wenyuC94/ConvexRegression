
function generate_examples(func_type,n,d,SNR,random_state =42)
    srand(random_state);
    x = 2*rand(n,d) - 1;
    for i in 1:d
        x[:,i] = x[:,i] - mean(x[:,i]);
        x[:,i] = x[:,i] / norm(x[:,i],2);
    end
    
    y = zeros(n,1);
    f = zeros(n,1);
    nse = randn(n,1);
    xi = zeros(n,d);
    
    if func_type == "A"
        ######## Type A ########
        ### phi(x) = norm(x,2)^2
        for i in 1:n
            f[i] = norm(x[i,:],2)^2;
            xi[i,:] = 2*x[i,:];
        end
    elseif func_type == "B"
        ######## Type B ########
        ### phi(x) = norm(x,2)+x_1^2+x_2^2
        for i in 1:n
            f[i] = norm(x[i,:],2)+x[i,1]^2+x[i,2]^2;
            xi[i,:] = x[i,:]/norm(x[i,:],2);
            xi[i,1:2] += 2*x[i,1:2]
        end        
    elseif func_type == "C"
        ######## Type C ########
        ### phi(x) = norm(x[4:],Inf)+norm(x[i,1:3],2)^2
        for i in 1:n
            f[i] = 5*norm(x[i,4:end],Inf) + norm(x[i,1:3],2)^2 ;
            xi[i,1:3] = 2*x[i,1:3];
            tmp = abs.(x[i,4:end]).==norm(x[i,4:end],Inf);
            xi[i,4:end] = 5*tmp/sum(tmp) .* sign.(x[i,4:end]);
        end 
    elseif func_type == "D"
        ######## Type D ########
        ### phi(x) = max_j <a_j,x>
        a = 2 * rand(d,3) - 1
        for i in 1:n
            f[i],j = findmax([a[:,j]'*x[i,:] for j = 1:3])
            xi[i,:] = a[:,j]
        end
    elseif func_type == "E"
        ######## Type E ########
        ### phi(x) = 5norm(x,Inf) + norm(x,2)^2
        for i in 1:n
            f[i] = 5*norm(x[i,:],Inf) + norm(x[i,:],2)^2;
            xi[i,:] = 2*x[i,:];
            tmp = abs.(x[i,:]).==norm(x[i,:],Inf);
            xi[i,:] += 5*tmp/sum(tmp).* sign.(x[i,:]);
        end
    end
                    
    f = f - mean(f);
    tau = var(f) / (SNR*var(nse));
    nse = nse * sqrt(tau);
    y = f + nse;
    norm_y = norm(y);
    y /= norm_y;
    f /= norm_y;
    xi /= norm_y;
    
    return x,y,f,xi
end
