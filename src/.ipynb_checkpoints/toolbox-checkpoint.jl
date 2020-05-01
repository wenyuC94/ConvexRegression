function ptoi(i,j,n)
    return (i-1)*(n-1)+((j>i)?(j-1):j)
end
function itop(idx,n)
    i = Int(floor((idx-1)/(n-1)))+1;
    j = (idx-1)%(n-1)+1;
    j += (j>=i);
    return (i,j)
end
function A_dot_s(W,s,tmp=[])
    tmp = zeros(length(W),1);
    idx = 1;
    for (i,j) in W
        tmp[idx] = s[j] - s[i];
        idx += 1;
    end
    return tmp
end
# function A_dot_s(W,s,tmp=[])
#     N = length(W)
#     tmp = SharedArray{Float64}(N);
#     idx = 1;
#     @parallel for idx in 1:N
#         j = W[idx][2];
#         i = W[idx][1];
#         tmp[idx] = s[j] - s[i];
#     end
#     return reshape(Array(tmp),(length(W),1))
# end

function AT_dot_lamb(W,lamb,n,tmp =[])
    tmp = zeros(n,1);
    idx = 1;
    for (i,j) in W
        tmp[i] -= lamb[idx];
        tmp[j] += lamb[idx];
        idx += 1;
    end
    return tmp
end


function B_dot_s(Xmat,W,s,tmp = [])
        tmp = zeros(length(W),1);
        n,d = size(Xmat);
        idx = 1;
        for (i,j) in W
            tmp[idx] = (Xmat[i,:] - Xmat[j,:])'*s[((i-1)*d+1):(i*d)];
            idx += 1;
        end
        return tmp
end


function BT_dot_lamb(Xmat,W,lamb,tmp = [])
        n,d = size(Xmat);
        tmp = zeros(n*d,1);
        idx = 1;
        for (i,j) in W
                tmp[((i-1)*d+1):(i*d)] += (Xmat[i,:] - Xmat[j,:]) * lamb[idx];
                idx += 1;
        end
        return tmp
end


function S_dot_lamb(Xmat,rho,W,lamb,if_substract_Y=0,Y=[])
    n = size(Xmat,1);
    if if_substract_Y==0
        Y = zeros(n,1);
    end
    tmp = A_dot_s(W,AT_dot_lamb(W,lamb,n)-Y)+1.0/rho*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,lamb));
    return tmp;
end

function power_iter(Xmat,W,rho,maxiter = 300,TOL=1e-3)
    rng = MersenneTwister(1234);
    x = randn(rng,Float64,(length(W),1));
    x /= norm(x);
    y = zeros(length(W),1); # pre-alloc
    nm = 0;
    n,d = size(Xmat);
    #tmp_W = zeros(length(W),1);
    #tmp_n = zeros(n,1);
    #tmp_nd = zeros(n*d,1);
    for k in 1:maxiter
        #y = A_dot_s(W,AT_dot_lamb(W,x,n,tmp_n),tmp_W)+1.0/rho*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,x,tmp_nd),tmp_W);
        y = S_dot_lamb(Xmat,rho,W,x);
        if (abs(nm-norm(y))<TOL)
                break;
        end
        nm = norm(y);
        x = y/nm;
    end

    return nm;
end

function generate_quadratic(Xmat,Y,rho)
    n = size(Xmat,1);
    S = zeros(n*(n-1),n*(n-1));
    W = Array{Tuple{Int64,Int64},1}(n*(n-1));
    idx = 1;
    for i in 1:n
        for j in setdiff(1:n,i)
            W[idx] = (i,j);
            idx +=1;
        end
    end

    for k in 1:n*(n-1)
        ek = zeros(n*(n-1),1);
        ek[k] = 1;
        S[:,k] = S_dot_lamb(Xmat,rho,W,ek);
    end
    s = A_dot_s(W,Y);
    return S,s;
end


function get_objective_value(Xmat, Y, rho, W, lamb)
    n = size(Xmat,1);
    obj = 0.5*lamb'*A_dot_s(W,AT_dot_lamb(W,lamb,n)-2Y)+1/(2*rho)*lamb'*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,lamb));
    return obj
end

function get_primal_solution_partial(Xmat,Y,rho,W,lamb,ifsummary=false)
    n,d = size(Xmat);
    phi = zeros(n,1);
    xi = zeros(n*d,1);
    phi = Y - AT_dot_lamb(W,lamb,n);
    xi = -BT_dot_lamb(Xmat,W,lamb)/rho;
    if (ifsummary)
        lamb = lamb[1:length(W)];
        obj = get_objective_value(Xmat, Y, rho, W, lamb);
        Wlen = sum(lamb.!=0);
        W = W[lamb.!=0];
        lamb = lamb[lamb.!=0];
        Omega = Array{Tuple{Int64,Int64},1}(n*(n-1));
        idx = 1;
        for i in 1:n
            for j in setdiff(1:n,i)
                Omega[idx] = (i,j);
                idx +=1;
            end
        end
        viol = A_dot_s(Omega,phi) + B_dot_s(Xmat,Omega,xi);
        return phi,xi,lamb,viol,obj,W,Wlen
    end
    return phi,xi
end

function get_primal_solution_whole(Xmat,Y,rho,lamb)
    n = size(Xmat,1);
    W = Array{Tuple{Int64,Int64},1}(n*(n-1));
    idx = 1;
    for i in 1:n
        for j in setdiff(1:n,i)
            W[idx] = (i,j);
            idx +=1;
        end
    end
    phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
    viol = A_dot_s(W,phi) + B_dot_s(Xmat,W,xi);
    obj = get_objective_value(Xmat, Y, rho, W, lamb);
    Wlen = sum(lamb.!=0);
    W = W[(lamb.!=0)[:]];
    lamb = lamb[(lamb.!=0)[:]];
    return phi,xi,lamb,viol,obj,W,Wlen
end

function get_primal_feasible_solution(Xmat,Y,phi,xi,TOL = 1.0e-2)
    n,d = size(Xmat);
    phi_f = zeros(n,1);
    xi_f = zeros(n*d,1);
    for j in 1:n
        W = [(i,j) for i in 1:n];
        viol_j = A_dot_s(W,phi) + B_dot_s(Xmat,W,xi);
        viol_j[j] = 0;
        max_viol = minimum(viol_j);
        phi_f[j] = phi[j] - max_viol;
        xi_norm = 1e10;
        for i in 1:n
            if viol_j[i] == max_viol
                if vecnorm(xi[(i-1)*d+1:i*d],2) < xi_norm
                    xi_f[(j-1)*d+1:j*d] = xi[(i-1)*d+1:i*d]
                end
            end
        end
    end
    return phi_f,xi_f;
end

function check_feasibility(Xmat,Y, phi,xi, TOL = 1.e-10)
    n = size(Xmat,1);
    Omega = Array{Tuple{Int64,Int64},1}(n*(n-1));
    idx = 1;
    for i in 1:n
        for j in setdiff(1:n,i)
            Omega[idx] = (i,j);
            idx +=1;
        end
    end
    viol = A_dot_s(Omega,phi) + B_dot_s(Xmat,Omega,xi);
    #println(minimum(viol))
    return minimum(viol) >= -TOL;
end

function get_duality_gap(Xmat, Y, rho, lamb, TOL = 1.0e-2)
    #println(length(lamb));
    phi,xi,lamb,viol,L_upper,W,Wlen = get_primal_solution_whole(Xmat,Y,rho,lamb);
    phi_f,xi_f = get_primal_feasible_solution(Xmat,Y,phi,xi,1.0e-2);
    L_lower = 0.5*vecnorm(phi_f-mean(phi_f)-Y+mean(Y))^2+rho/2*vecnorm(xi_f)^2;
    println(check_feasibility(Xmat,Y,phi_f,xi_f));
    max_viol = -minimum(viol);
    num_tol_viol = sum(viol.<-TOL);
    num_viol = sum(viol.<0);
    return phi, xi, lamb, L_upper, L_upper-L_lower, max_viol, num_tol_viol, num_viol, W, Wlen
end

function get_duality_gap_partial(Xmat, Y, rho, W, lamb, TOL = 1.0e-2)
    phi,xi,lamb,viol,L_upper,W,Wlen = get_primal_solution_partial(Xmat,Y,rho,W,lamb,true)
    phi_f,xi_f = get_primal_feasible_solution(Xmat,Y,phi,xi,1.0e-2);
    L_lower = 0.5*vecnorm(phi_f-Y)^2+rho/2*vecnorm(xi_f)^2;
    println(check_feasibility(Xmat,Y,phi_f,xi_f));
    max_viol = -minimum(viol);
    num_tol_viol = sum(viol.<-TOL);
    num_viol = sum(viol.<0);
    return phi, xi, lamb, L_upper, L_upper+L_lower, max_viol, num_tol_viol, num_viol, W, Wlen
end

function get_duality_gap_partial_info(Xmat, Y, rho, W, lamb, TOL = 1.0e-2)
    phi,xi,lamb,viol,L_upper,W,Wlen = get_primal_solution_partial(Xmat,Y,rho,W,lamb,true)
    phi_f,xi_f = get_primal_feasible_solution(Xmat,Y,phi,xi,1.0e-2);
    L_lower = 0.5*vecnorm(phi_f-mean(phi_f)-Y+mean(Y))^2+rho/2*vecnorm(xi_f)^2;
    #println(check_feasibility(Xmat,Y,phi_f,xi_f));
    max_viol = -minimum(viol);
    num_tol_viol = sum(viol.<-TOL);
    num_viol = sum(viol.<0);
    return L_upper+L_lower, max_viol, num_tol_viol, num_viol, Wlen, phi_f, xi_f
end

function partition_n_by_m(n,m)
    np = Int(floor(n/m));
    if np == n/m
        partition = [(i-1)*m+1:i*m for i in 1:np]
    else
        partition = [(i-1)*m+1:min(i*m,n) for i in 1:(np+1)]
    end
    return partition
end


function decoding_by_samples(W,Wlen,inv_samples)
    N = length(inv_samples);
    Wnew = Array{Tuple{Int64,Int64},1}(length(W));
    for k in 1:Wlen
        Wnew[k] = (inv_samples[W[k][1]],inv_samples[W[k][2]]);
        if inv_samples[W[k][1]] == 0 || inv_samples[W[k][2]] == 0
            println("wrong")
        end
    end
    Wnew[Wlen+1:end] = W[Wlen+1:end];
    return Wnew
end
    
function cvxreg_qp_ineq_APG(Xmat, Y, rho, Wwhole, Wlen, lamb, max_steps, TOL)
    n,d =size(Xmat);
    W = Wwhole[1:Wlen];
    theta = 1;
    L = power_iter(Xmat,W,rho);
        #println(L);
    lx = lamb[1:Wlen];
    ly = lx;
    fnew = 0;
    objs = zeros(max_steps+1);
    f = get_objective_value(Xmat, Y, rho, W, lx);
    k = 0;
    tmp = zeros(Wlen,1);
    lxnew = zeros(Wlen,1);
    for k in 1:max_steps
        tmp = S_dot_lamb(Xmat,rho,W,ly,1,Y);
        lxnew = min.(ly - tmp/L,0);
        fnew = get_objective_value(Xmat, Y, rho, W, lxnew);
        if (fnew[1,1] > f[1,1])
                theta = 1;
        end
        if (fnew[1,1] - f[1,1] >-TOL*0.1 && fnew[1,1] <= f[1,1] &&fnew[1,1]>f[1,1]*(1+TOL))
                break
        end
        thetanew = (1+sqrt(1+4*theta^2))/2;
        gam = (theta-1)/thetanew;
        ly = lxnew + gam*(lxnew - lx);
        lx = lxnew;
        theta = thetanew;
        f = fnew;
    end
    lamb[1:Wlen] = lxnew;
    phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
    return phi,xi,lamb,fnew[1,1]
end
    
function cvxreg_qp_ineq_APG_timing(Xmat, Y, rho, Wwhole, Wlen, lamb, max_steps, TOL)
    all_times = Array{Float64}([])
    cur_time = @elapsed begin
        n,d =size(Xmat);
        W = Wwhole[1:Wlen];
        theta = 1;
        #println(W)
        pow_time = @elapsed L = power_iter(Xmat,W,rho);
        #println(L);
        lx = lamb[1:Wlen];
        ly = lx;
        fnew = 0;
        objs = zeros(max_steps+1);
        f = get_objective_value(Xmat, Y, rho, W, lx);
        objs[1] = f[1,1]
        k = 0;
        tmp = zeros(Wlen,1);
        lxnew = zeros(Wlen,1);
    end
    append!(all_times,cur_time)
    for k in 1:max_steps
        cur_time += @elapsed begin
            #tmp = A_dot_s(W,AT_dot_lamb(W,ly,n,tmp_n)-Y,tmp_W)+1.0/rho*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,ly,tmp_nd),tmp_W);
            tmp = S_dot_lamb(Xmat,rho,W,ly,1,Y);
            lxnew = min.(ly - tmp/L,0);
            fnew = get_objective_value(Xmat, Y, rho, W, lxnew);
            objs[k+1] = fnew[1,1]
        end
        append!(all_times,cur_time)
        cur_time += @elapsed begin
            if (fnew[1,1] > f[1,1])
                    theta = 1;
            end
            if (fnew[1,1] - f[1,1] >-TOL && fnew[1,1] <= f[1,1] &&fnew[1,1]>f[1,1]*(1+TOL))
                    break
            end
            thetanew = (1+sqrt(1+4*theta^2))/2;
            gam = (theta-1)/thetanew;
            ly = lxnew + gam*(lxnew - lx);
            lx = lxnew;
            theta = thetanew;
            f = fnew;
        end
    end
        lamb[1:Wlen] = lxnew;
        phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
    return objs[1:k+1],all_times[1:k+1],pow_time,phi,xi,lamb
end
    
    
function cvxreg_qp_ineq_APG_ls(Xmat, Y, rho, Wwhole, Wlen, lamb, max_steps, L, TOL=1e-10)
        beta_1 = 0.9
        beta_2 = 0.98
            n,d =size(Xmat);
            W = Wwhole[1:Wlen];
            theta = 1;
            
            search_flag = 0;
            if L <= 0
                L = power_iter(Xmat,W,rho);
                pow_time = 1;
            else
                pow_time = -1;
            end
            lx = lamb[1:Wlen];
            ly = lx;
            fnew = 0;
            objs = zeros(max_steps+1);
            f = get_objective_value(Xmat, Y, rho, W, lx);
            objs[1] = f[1,1]
            k = 0;
            tmp = zeros(Wlen,1);
            lxnew = zeros(Wlen,1);
        for k in 1:max_steps
                    #tmp = A_dot_s(W,AT_dot_lamb(W,ly,n,tmp_n)-Y,tmp_W)+1.0/rho*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,ly,tmp_nd),tmp_W);
                    tmp = S_dot_lamb(Xmat,rho,W,ly,1,Y);
                    while(true)
                        lxnew = min.(ly - tmp/L,0);
                        fnew = get_objective_value(Xmat, Y, rho, W, lxnew);
                        if (pow_time > 0 || fnew[1,1] <= f[1,1] + (tmp' * (lxnew - ly))[1,1] + L/2*norm(lxnew-ly,2)^2 || search_flag >= 50)
                            search_flag = 0
                            break
                        else
                            L /= ((k==1)? beta_1:beta_2);
                            search_flag += 1
                            if search_flag >= 20
                                L = power_iter(Xmat,W,rho,50)
                            end
                        end
                    end
                    objs[k+1] = fnew[1,1]
                    if (fnew[1,1] > f[1,1])
                            theta = 1;
                    end
                    if (fnew[1,1] - f[1,1] >-TOL && fnew[1,1] <= f[1,1])
                            break
                    end
                    thetanew = (1+sqrt(1+4*theta^2))/2;
                    gam = (theta-1)/thetanew;
                    ly = lxnew + gam*(lxnew - lx);
                    lx = lxnew;
                    theta = thetanew;
                    f = fnew;
        end
        lamb[1:Wlen] = lxnew;
        phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
        return phi,xi,lamb,L,fnew[1,1]
end
            
function cvxreg_qp_ineq_APG_ls_timing(Xmat, Y, rho, Wwhole, Wlen, lamb, max_steps, L, TOL=1e-10)
        beta_1 = 0.9
        beta_2 = 0.98
        all_times = Array{Float64}([])
        cur_time = @elapsed begin
            n,d =size(Xmat);
            W = Wwhole[1:Wlen];
            theta = 1;
            search_flag = 0;
            if L <= 0
                pow_time = @elapsed L = power_iter(Xmat,W,rho);
            else
                pow_time = -1
            end
            println("L = ", L);
            lx = lamb[1:Wlen];
            ly = lx;
            fnew = 0;
            objs = zeros(max_steps+1);
            f = get_objective_value(Xmat, Y, rho, W, lx);
            objs[1] = f[1,1]
            k = 0;
            tmp = zeros(Wlen,1);
            lxnew = zeros(Wlen,1);
            #tmp_W = zeros(Wlen,1);
            #tmp_n = zeros(n,1);
            #tmp_nd = zeros(n*d,1);
        end
        append!(all_times,cur_time)
        for k in 1:max_steps
                cur_time += @elapsed begin
                    #tmp = A_dot_s(W,AT_dot_lamb(W,ly,n,tmp_n)-Y,tmp_W)+1.0/rho*B_dot_s(Xmat,W,BT_dot_lamb(Xmat,W,ly,tmp_nd),tmp_W);
                    tmp = S_dot_lamb(Xmat,rho,W,ly,1,Y);
                    while(true)
                        lxnew = min.(ly - tmp/L,0);
                        fnew = get_objective_value(Xmat, Y, rho, W, lxnew);
                        if (pow_time > 0 || fnew[1,1] <= f[1,1] + (tmp' * (lxnew - ly))[1,1] + L/2*norm(lxnew-ly,2)^2 || search_flag >= 50)
                            search_flag = 0
                            break
                        else
                            L /= ((k==1)? beta_1:beta_2);
                            search_flag += 1
                            if search_flag >= 20
                                L = power_iter(Xmat,W,rho,50)
                            end
                        end
                    end
                    objs[k+1] = fnew[1,1]
                end
                append!(all_times,cur_time)
                cur_time += @elapsed begin
                    if (fnew[1,1] > f[1,1])
                            theta = 1;
                    end
                    if (fnew[1,1] - f[1,1] >-TOL && fnew[1,1] <= f[1,1])
                            break
                    end
                    thetanew = (1+sqrt(1+4*theta^2))/2;
                    gam = (theta-1)/thetanew;
                    ly = lxnew + gam*(lxnew - lx);
                    lx = lxnew;
                    theta = thetanew;
                    f = fnew;
                end
        end
        lamb[1:Wlen] = lxnew;
        phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
        return objs[1:k+1],all_times[1:k+1],pow_time,phi,xi,lamb,L
end