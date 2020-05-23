
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
        ### phi(x) = norm(x,4)^4
        for i in 1:ntrain
            f[i] = norm(x[i,:],4)^4;
            xi[i,:] = 4*x[i,:].^3;
        end
    elseif func_type == "C"
        ######## Type C ########
        ### phi(x) = max_j <a_j,x>
        a = 2 * rand(d,2*d) - 1
        for i in 1:ntrain
            f[i],j = findmax([a[:,j]'*x[i,:] for j = 1:2*d])
            xi[i,:] = a[:,j]
        end
    elseif func_type == "D"
        ######## Type D ########
        ### phi(x) =  <a,x>
        a = 2*rand(d,1) - 1
        for i in 1:ntrain
            f[i] = (a'*x[i,:])[1,1]
            xi[i,:] = a
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
