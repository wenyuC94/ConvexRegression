include("toolbox.jl")

function initial_active_set(Xmat,Y)
    n = size(Xmat,1);
    W = ifelse.(Y[1:n-1].>Y[2:n],[(i,i+1) for i in 1:n-1],[(i+1,i) for i in 1:n-1])
    return W
end



function augmentation_rule1(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,block=0)
    n,d = size(Xmat)
    Wtmp = Array{Tuple{Int64,Int64},1}(n-1);
    viol = zeros(n-1,1);
    viol_flags = zeros(n-1,1);
    Delta = Array{Tuple{Int64,Int64},1}(0);
    P1 = 0;
    p = Array{Int64,1}(P);
    for i in 1:n
        if block == 0
            Wtmp = setdiff([(i,j) for j in setdiff(1:n,i)],W);
        else
            Wtmp = setdiff([(j,i) for j in setdiff(1:n,i)],W);
        end
        viol = A_dot_s(Wtmp,phi)+B_dot_s(Xmat,Wtmp,xi);
        viol_flags = viol.<-TOL;
        P1 = min(sum(viol_flags),P);
        if P1 == 0
           continue
        elseif P1 == 1
           _,p = findmin(viol);
            push!(Delta,Wtmp[p])
        else
           p = sortperm(viol[:];alg=PartialQuickSort(P1))[1:P1]
           append!(Delta,Wtmp[p])
        end
    end
    return Delta
end

function augmentation_rule1_heuristic(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,J,Wcheck,Wchecklen,is_sort_iter,block=0)
    n,d = size(Xmat)
    p = Array{Int64,1}(P);
    Wtmp = Array{Tuple{Int64,Int64},1}(J);
    viol = zeros(J,1);
    Delta = Array{Tuple{Int64,Int64},1}(0);
    P1 = 0;
    p = Array{Int64,1}(P);
    flag = 0;
    for i in 1:n
        if is_sort_iter == false
            if Wchecklen[i] != 0 
                Wtmp = setdiff(Wcheck[i],W);
                if length(Wtmp) == 0
                    flag += 1;
                    continue
                end
                viol = A_dot_s(Wtmp,phi)+B_dot_s(Xmat,Wtmp,xi);
                viol_flags = viol.<-TOL;
                P1 = min(sum(viol_flags),P);
                if P1 == 0
                   continue
                elseif P1 == 1
                   _,p = findmin(viol);
                    push!(Delta,Wtmp[p])
                else
                   p = sortperm(viol[:];alg=PartialQuickSort(P1))[1:P1]
                   append!(Delta,Wtmp[p])
                end
            else
                flag += 1;
            end
        else
            if block == 0
                Wtmp = setdiff([(i,j) for j in setdiff(1:n,i)],W);
            else
                Wtmp = setdiff([(j,i) for j in setdiff(1:n,i)],W);
            end 
            viol = A_dot_s(Wtmp,phi)+B_dot_s(Xmat,Wtmp,xi);
            ntmp = min(length(viol),J)
            p = sortperm(viol[:];alg=PartialQuickSort(ntmp))[1:ntmp]
            Wtmp = Wtmp[p];
            viol = viol[p];
            if viol[1] >= -TOL
                Wchecklen[i] = 0;
                flag += 1;
            else
                viol_flags = viol.<-TOL
                Wchecklen[i] = sum(viol_flags)
                Wcheck[i] = Wtmp[viol_flags]
                append!(Delta,Wcheck[i][1:min(P,Int(Wchecklen[i]))])
            end
        end
    end
    if (flag == n)
        if is_sort_iter == false
            return augmentation_rule1_heuristic(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,J,Wcheck,Wchecklen,true,block)
        else
            return Delta
        end
    else
        return Delta
    end
end


function augmentation_rule2(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,K)
    n,d = size(Xmat);
    all_i = rand(1:n,K)
    all_j = rand(1:n,K)
    Delta = [(i,j) for (i,j) in zip(all_i,all_j) if i != j ]
    Delta = unique(setdiff(Delta,W[1:Wlen]))
    viol = A_dot_s(Delta, phi) + B_dot_s(Xmat, Delta,xi);
    Delta = Delta[viol[:].<-TOL];
    return Delta
end

function augmentation_rule3(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,block=0)
    n,d = size(Xmat);
    if block ==0
        Delta = [(i,j) for i in 1:n for j in rand(setdiff(1:n,i),P)]
    else
        Delta = [(j,i) for i in 1:n for j in rand(setdiff(1:n,i),P)]
    end
    Delta = unique(setdiff(Delta,W[1:Wlen]))
    viol = A_dot_s(Delta, phi) + B_dot_s(Xmat, Delta,xi);
    Delta = Delta[viol[:].<-TOL];
    return Delta
end

function augmentation_rule4(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,M,K)
    n,d = size(Xmat);
    all_i = rand(1:n,M);
    Delta = [(i,rand(setdiff(1:n,i))) for i in all_i];
    Delta = unique(setdiff(Delta,W[1:Wlen]))
    viol = A_dot_s(Delta,phi)+B_dot_s(Xmat,Delta,xi);
    viol_flags = viol.<-TOL;
    P1 = min(sum(viol_flags),K);
    if P1 == 0
        return Array{Tuple{Int64,Int64},1}(0);
    elseif P1 == 1
        _,p = findmin(viol)
        return [Delta[p]]
    else
        p = sortperm(viol[:];alg=PartialQuickSort(P1))[1:P1]
        return Delta[p]
    end
end

function augmentation_rule5(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,G,P,block=0)
    n,d = size(Xmat)
    Wtmp = Array{Tuple{Int64,Int64},1}(n-1);
    viol = zeros(n-1,1);
    Delta = Array{Tuple{Int64,Int64},1}(0);
    P1 = 0;
    for i in rand(1:n,G)
        if block == 0
            Wtmp = setdiff([(i,j) for j in setdiff(1:n,i)],W);
        else
            Wtmp = setdiff([(j,i) for j in setdiff(1:n,i)],W);
        end
        viol = A_dot_s(Wtmp,phi)+B_dot_s(Xmat,Wtmp,xi);
        viol_flags = viol.<-TOL;
        P1 = min(sum(viol_flags),P);
        if P1 == 0
           continue
        elseif P1 == 1
           _,p = findmin(viol);
            push!(Delta,Wtmp[p])
        else
           p = sortperm(viol[:];alg=PartialQuickSort(P1))[1:P1]
           append!(Delta,Wtmp[p])
        end
    end
    return Delta
end

function active_set_augmentation(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,rule,P,M,K,G,block,J,Wcheck,Wchecklen)
    if rule == 1
        return augmentation_rule1(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,block)
    elseif rule == -1
        return augmentation_rule1_heuristic(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,J,Wcheck,Wchecklen,false,block)
    elseif rule == 2
        return augmentation_rule2(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,K)
    elseif rule == 3
        return augmentation_rule3(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,P,block)
    elseif rule == 4
        return augmentation_rule4(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,M,K)
    elseif rule == 5
        return augmentation_rule5(Xmat,Y,W,Wlen,phi,xi,lamb,TOL,G,P,block)
    elseif rule == 0
        return initial_active_set(Xmat,Y)
    end
end

function warm_start(Xmat,Y,rho,partition_method,max_steps = 100,maxiter=20,violTOL=1.0e-2,innerTOL=1.0e-3,outerTOL = 1.0e-3,K=100,block=0)
    n,d = size(Xmat);
    W = Array{Tuple{Int64,Int64},1}(n*(n-1));
    lamb = zeros(n*(n-1));
    Wlen = 0;
    Wlen = 0;
    obj = 0;
    if partition_method == "thousand"
        partition = partition_n_by_m(n,1000);
        for prtt in partition
            _,_,lambtmp, Wtmp,Wlentmp,objtmp = active_set(Xmat[prtt,:], Y[prtt], rho, 5,50,1.0e-3, 1.0e-4, 1.0e-5, 0,-1,true, false, true,2,"five",1,0,K,0,block)
            obj+= objtmp;
            Wlennew = Wlen + sum(lambtmp.!=0);
            W[Wlen+1:Wlennew] = [(prtt[i],prtt[j]) for (i,j) in Wtmp[(lambtmp.!=0)[:]]];
            lamb[Wlen+1:Wlennew] = lambtmp[lambtmp.!=0];
            Wlen = Wlennew;
        end
        return W, lamb, Wlen,obj;
    end
    num_dict = Dict("five"=>5,"ten"=>10,"fifteen"=>15,"twenty"=>20,"fifty"=>50)
    if partition_method in keys(num_dict)
        m = cld(n,num_dict[partition_method]);
        partition = partition_n_by_m(n,m);
        for prtt in partition
            _,_,lambtmp, Wtmp,Wlentmp,objtmp = active_set(Xmat[prtt,:], Y[prtt], rho, 5,50,1.0e-3, 1.0e-4, 1.0e-5, 0,-1,true, false, true,2,"five",1,0,K,0,block)
            obj+= objtmp;
            Wlennew = Wlen + sum(lambtmp.!=0);
            W[Wlen+1:Wlennew] = [(prtt[i],prtt[j]) for (i,j) in Wtmp[(lambtmp.!=0)[:]]];
            lamb[Wlen+1:Wlennew] = lambtmp[lambtmp.!=0];
            Wlen = Wlennew;
        end
        return W, lamb, Wlen,obj;
    end
end


function warm_start_limited_memory(Xmat,Y,rho,partition_method,max_steps = 100,maxiter=20,violTOL=1.0e-2,innerTOL=1.0e-3,outerTOL = 1.0e-3,K=100,block=0)
    n,d = size(Xmat);
    W = Array{Tuple{Int64,Int64},1}(0);
    lamb = zeros(0);
    Wlen = 0;
    obj = 0;
    if partition_method == "thousand"
        partition = partition_n_by_m(n,1000);
        for prtt in partition
            _,_,lambtmp, Wtmp,Wlentmp,objtmp = active_set_limited_memory(Xmat[prtt,:], Y[prtt], rho, 5,50,1.0e-3, 1.0e-4, 1.0e-5, 0,-1,true, false, true,2,"five",1,0,K,0,block)
            obj+= objtmp;
            Wtmp = [(prtt[i],prtt[j]) for (i,j) in Wtmp[(lambtmp.!=0)[:]]];
            lambtmp = lambtmp[lambtmp.!=0];
            append!(W,Wtmp);
            append!(lamb,lambtmp);
            Wlen += length(Wtmp);
        end
        return W, lamb, Wlen,obj;
    end
    num_dict = Dict("five"=>5,"ten"=>10,"fifteen"=>15,"twenty"=>20,"fifty"=>50)
    if partition_method in keys(num_dict)
        m = cld(n,num_dict[partition_method]);
        partition = partition_n_by_m(n,m);
        for prtt in partition
            _,_,lambtmp, Wtmp,Wlentmp,objtmp = active_set_limited_memory(Xmat[prtt,:], Y[prtt], rho, 5,50,1.0e-3, 1.0e-4, 1.0e-5, 0,-1,true, false, true,2,"five",1,0,K,0,block)
            obj+= objtmp;
            Wtmp = [(prtt[i],prtt[j]) for (i,j) in Wtmp[(lambtmp.!=0)[:]]];
            lambtmp = lambtmp[lambtmp.!=0];
            append!(W,Wtmp);
            append!(lamb,lambtmp);
            Wlen += length(Wtmp);
        end
        return W, lamb, Wlen,obj;
    end
end


function active_set_timing_limited_memory(Xmat, Y, rho, max_steps= 5,maxiter=50,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, verbose = 1,random_state = 42,maxtime = 10800,LS =true, WS=false,inexact = true,rule=2,warm_start_partition="five",P=1,M=0,K=100,G=0,block=0,heuristics=false,J=0)
    tic();
    if random_state != -1
        srand(random_state);
    end
    if WS == false
        cur_time = @elapsed begin
            searching_times = Array{Float64}([])
            all_obj_times = Array{Float64}([])
            pow_times = Array{Float64}([])
            optimization_starting_times = Array{Int}([])
            n,d = size(Xmat);
            W = Array{Tuple{Int64,Int64},1}(0);
            Wlen = 0;
            lamb = zeros(0,1);
            phi = Y;
            xi = zeros(n*d,1);
            k = 0;
            flag = 0;
            all_objs = Array{Float64}([])
            if heuristics == true
                rule = -1
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
                Wchecklen = Array{Int64,1}(n)
                for i in 1:n
                    Wchecklen[i] = 0;
                    Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
                end
            else
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
                Wchecklen = Array{Int64,1}(0)
            end
        end
        initialize_time = cur_time;
        searching_time = 0;
    else
        cur_time = @elapsed begin
            searching_times = Array{Float64}([])
            all_obj_times = Array{Float64}([])
            pow_times = Array{Float64}([])
            optimization_starting_times = Array{Int}([])
            n,d = size(Xmat);
            W,lamb,Wlen,last_obj = warm_start_limited_memory(Xmat,Y,rho,warm_start_partition,100,20,1.0e-2,1.0e-3,1.0e-3,100,block);
            sorted_idx = sortperm(W);
            W = W[sorted_idx];
            lamb = lamb[sorted_idx];
            phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
            all_objs = Array{Float64}([])
            L = 0;
            if heuristics == true
                rule = -1
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
                Wchecklen = Array{Int64,1}(n)
                for i in 1:n
                    Wchecklen[i] = 0;
                    Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
                end
            else
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
                Wchecklen = Array{Int64,1}(0)
            end
        end
        initialize_time = cur_time;
        searching_time = 0;
        append!(all_objs, last_obj)
        append!(all_obj_times,cur_time)
    end
    if LS == true
        L = 0
    end
    
    for k in 1:maxiter
        searching_time = @elapsed begin
            Delta = active_set_augmentation(Xmat,Y,W,Wlen,phi,xi,lamb,violTOL,rule,P,M,K,G,block,J,Wcheck,Wchecklen);
            Delta_len = length(Delta);
            if verbose >= 1
                println("Iteration:")
                println(k);
            end
            if Delta_len != 0
                append!(W,Delta)
                lamb = vcat(lamb,zeros(Delta_len,1))
                sorted_idx = sortperm(W);
                W = W[sorted_idx];
                lamb = lamb[sorted_idx];
                flag = 0
                if verbose >= 1
                    println(Wlen)
                end
                Wlen = Wlen + Delta_len
                if verbose >= 1
                    println(Wlen)
                end
            else
                flag += 1
                if verbose >= 1
                    println(Wlen)
                    println(Wlen)
                end
            end
        end
        cur_time += searching_time
        append!(searching_times,searching_time)
        tmp_steps = max_steps
        if inexact == true
            if  k == maxiter
                tmp_steps = 100
            else
                tmp_steps = max_steps
            end
        end
        append!(optimization_starting_times, length(all_objs)+1)
        if LS == true
            solvetime = @elapsed objs,times,pow_time,phi,xi,lamb,L = cvxreg_qp_ineq_APG_ls_timing(Xmat, Y, rho, W, Wlen, lamb, tmp_steps, L,innerTOL)
        else
            solvetime = @elapsed objs,times,pow_time,phi,xi,lamb = cvxreg_qp_ineq_APG_timing(Xmat, Y, rho, W, Wlen, lamb, tmp_steps, innerTOL)
        end
        if cur_time + solvetime >= maxtime
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            break
        end
        if k > 1
            last_obj = all_objs[end];
        end
        if k > 1&& objs[end] - last_obj > -outerTOL*0.1 && objs[end] > last_obj * (1+outerTOL) && flag >= 5
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            cur_time += solvetime
            break
        else
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            cur_time += solvetime
        end
    end
    solvetime = toq();
    if verbose >= 1
        println(solvetime)
    end
    return all_objs,all_obj_times,initialize_time,searching_times,pow_times, optimization_starting_times,phi,xi,lamb, W,Wlen
end


function active_set_limited_memory(Xmat, Y, rho, max_steps= 5,maxiter=50,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, verbose = 1,random_state = 42,LS =true, WS=false,inexact = true,rule=1,warm_start_partition="five",P=1,M=0,K=100,G=0,block=0,heuristics=false,J=0)
    tic();
    if random_state != -1
        srand(random_state);
    end
    if WS == false
        n,d = size(Xmat);
        tic();
        W = Array{Tuple{Int64,Int64},1}(0);
        Wlen = 0;
        lamb = zeros(0,1);
        phi = Y;
        xi = zeros(n*d,1);
        flag = 0;
        obj = 0;
        last_obj = 0;
        if heuristics == true
            rule = -1
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
            Wchecklen = Array{Int64,1}(n)
            for i in 1:n
                Wchecklen[i] = 0;
                Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
            end
        else
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
            Wchecklen = Array{Int64,1}(0)
        end
    else
        n,d = size(Xmat);
        W,lamb,Wlen,last_obj = warm_start_limited_memory(Xmat,Y,rho,warm_start_partition,100,20,1.0e-2,1.0e-3,1.0e-3,100,block);
        sorted_idx = sortperm(W);
        W = W[sorted_idx];
        lamb = lamb[sorted_idx];
        phi,xi = get_primal_solution_partial(Xmat,Y,rho,W,lamb);
        obj = 0
        L = 0;
        if heuristics == true
            rule = -1
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
            Wchecklen = Array{Int64,1}(n)
            for i in 1:n
                Wchecklen[i] = 0;
                Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
            end
        else
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
            Wchecklen = Array{Int64,1}(0)
        end
    end
    if LS == true
        L = 0
    end
    
    for k in 1:maxiter
        Delta = active_set_augmentation(Xmat,Y,W,Wlen,phi,xi,lamb,violTOL,rule,P,M,K,G,block,J,Wcheck,Wchecklen);
        Delta_len = length(Delta);
        if verbose >= 1
            println("Iteration:")
            println(k);
        end
        if Delta_len != 0
            append!(W,Delta)
            lamb = vcat(lamb,zeros(Delta_len,1))
            sorted_idx = sortperm(W);
            W = W[sorted_idx];
            lamb = lamb[sorted_idx];
            flag = 0
            if verbose >= 1
                println(Wlen)
            end
            Wlen = Wlen + Delta_len
            if verbose >= 1
                println(Wlen)
            end
        else
            flag += 1
            if verbose >= 1
                println(Wlen)
                println(Wlen)
            end
        end
        tmp_steps = max_steps
        if inexact == true
            if  k == maxiter
                tmp_steps = 100
            else
                tmp_steps = max_steps
            end
        end
        if LS == true
            phi,xi,lamb,L,obj = cvxreg_qp_ineq_APG_ls(Xmat, Y, rho, W, Wlen, lamb, tmp_steps, L,innerTOL)
        else
            phi,xi,lamb,obj = cvxreg_qp_ineq_APG(Xmat, Y, rho, W, Wlen, lamb, max_steps, innerTOL)
        end
        if k > 1&& obj - last_obj > -outerTOL*0.1 && obj > last_obj * (1+outerTOL) && flag >= 5
            if verbose >= 1
                println(obj)
            end
            break
        else
            if verbose >= 1
                println(obj)
            end
            last_obj = obj;
        end
    end
    solvetime = toq();
    if verbose >= 1
        println(solvetime)
    end
    return phi,xi,lamb, W,Wlen,obj
end


function active_set_timing(Xmat, Y, rho, max_steps= 5,maxiter=50,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, verbose = 1,random_state = 42,maxtime = 10800,LS =true, WS=false,inexact = true,rule=2,warm_start_partition="five",P=1,M=0,K=100,G=0,block=0,heuristics=false,J=0)
    tic();
    if random_state != -1
        srand(random_state);
    end
    if WS == false
        cur_time = @elapsed begin
            searching_times = Array{Float64}([])
            all_obj_times = Array{Float64}([])
            pow_times = Array{Float64}([])
            optimization_starting_times = Array{Int}([])
            n,d = size(Xmat);
            W = Array{Tuple{Int64,Int64},1}(n*(n-1));
            Wlen = 0;
            lamb = zeros(n*(n-1),1);
            phi = Y;
            xi = zeros(n*d,1);
            k = 0;
            flag = 0;
            all_objs = Array{Float64}([])
            sorted_idx = Array{Int64}(n*(n-1));
            if heuristics == true
                rule = -1
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
                Wchecklen = Array{Int64,1}(n)
                for i in 1:n
                    Wchecklen[i] = 0;
                    Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
                end
            else
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
                Wchecklen = Array{Int64,1}(0)
            end
        end
        initialize_time = cur_time;
        searching_time = 0;
    else
        cur_time = @elapsed begin
            searching_times = Array{Float64}([])
            all_obj_times = Array{Float64}([])
            pow_times = Array{Float64}([])
            optimization_starting_times = Array{Int}([])
            n,d = size(Xmat);
            W,lamb,Wlen,last_obj = warm_start(Xmat,Y,rho,warm_start_partition,100,20,1.0e-2,1.0e-3,1.0e-3,100,block);
            sorted_idx = Array{Int64}(n*(n-1));
            sorted_idx[1:Wlen] = sortperm(W[1:Wlen]);
            W[1:Wlen] = W[sorted_idx[1:Wlen]];
            lamb[1:Wlen] = lamb[sorted_idx[1:Wlen]];
            phi,xi = get_primal_solution_partial(Xmat,Y,rho,W[1:Wlen],lamb);
            all_objs = Array{Float64}([])
            L = 0;
            if heuristics == true
                rule = -1
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
                Wchecklen = Array{Int64,1}(n)
                for i in 1:n
                    Wchecklen[i] = 0;
                    Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
                end
            else
                Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
                Wchecklen = Array{Int64,1}(0)
            end
        end
        initialize_time = cur_time;
        searching_time = 0;
        append!(all_objs, last_obj)
        append!(all_obj_times,cur_time)
    end
    if LS == true
        L = 0
    end
    
    for k in 1:maxiter
        searching_time = @elapsed begin
            Delta = active_set_augmentation(Xmat,Y,W[1:Wlen],Wlen,phi,xi,lamb,violTOL,rule,P,M,K,G,block,J,Wcheck,Wchecklen);
            Delta_len = length(Delta);
            if verbose >= 1
                println("Iteration:")
                println(k);
            end
            if Delta_len != 0
                Wlen_new = Wlen + Delta_len
                W[Wlen+1:Wlen_new] = Delta
                sorted_idx[1:Wlen_new] = sortperm(W[1:Wlen_new]);
                W[1:Wlen_new] = W[sorted_idx[1:Wlen_new]];
                lamb[1:Wlen_new] = lamb[sorted_idx[1:Wlen_new]];
                flag = 0
                if verbose >= 1
                    println(Wlen)
                end
                Wlen = Wlen_new
                if verbose >= 1
                    println(Wlen)
                end
            else
                flag += 1
                if verbose >= 1
                    println(Wlen)
                    println(Wlen)
                end
            end
        end
        cur_time += searching_time
        append!(searching_times,searching_time)
        tmp_steps = max_steps
        if inexact == true
            if  k == maxiter
                tmp_steps = 100
            else
                tmp_steps = max_steps
            end
        end
        append!(optimization_starting_times, length(all_objs)+1)
        if LS == true
            solvetime = @elapsed objs,times,pow_time,phi,xi,lamb,L = cvxreg_qp_ineq_APG_ls_timing(Xmat, Y, rho, W[1:Wlen], Wlen, lamb, tmp_steps, L,innerTOL)
        else
            solvetime = @elapsed objs,times,pow_time,phi,xi,lamb = cvxreg_qp_ineq_APG_timing(Xmat, Y, rho, W[1:Wlen], Wlen, lamb, tmp_steps, innerTOL)
        end
        if cur_time + solvetime >= maxtime
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            break
        end
        if k > 1
            last_obj = all_objs[end];
        end
        if k > 1&& objs[end] - last_obj > -outerTOL*0.1 && objs[end] > last_obj * (1+outerTOL) && flag >= 5
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            cur_time += solvetime
            break
        else
            append!(all_objs,objs)
            append!(all_obj_times,cur_time+times)
            append!(pow_times, pow_time)
            if verbose >= 1
                println(objs[end])
                println(all_obj_times[end])
            end
            cur_time += solvetime
        end
    end
    solvetime = toq();
    if verbose >= 1
        println(solvetime)
    end
    return all_objs,all_obj_times,initialize_time,searching_times,pow_times, optimization_starting_times,phi,xi,lamb, W,Wlen
end

function active_set(Xmat, Y, rho, max_steps= 5,maxiter=50,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, verbose = 1,random_state = 42,LS =true, WS=false,inexact = true,rule=1,warm_start_partition="five",P=1,M=0,K=100,G=0,block=0,heuristics=false,J=0)
    tic();
    if random_state != -1
        srand(random_state);
    end
    if WS == false
        n,d = size(Xmat);
        W = Array{Tuple{Int64,Int64},1}(n*(n-1));
        Wlen = 0;
        lamb = zeros(n*(n-1),1);
        phi = Y;
        xi = zeros(n*d,1);
        k = 0;
        flag = 0;
        sorted_idx = Array{Int64}(n*(n-1));
        obj = 0;
        last_obj = 0;
        if heuristics == true
            rule = -1
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
            Wchecklen = Array{Int64,1}(n)
            for i in 1:n
                Wchecklen[i] = 0;
                Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
            end
        else
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
            Wchecklen = Array{Int64,1}(0)
        end
    else
        n,d = size(Xmat);
        W,lamb,Wlen,last_obj = warm_start(Xmat,Y,rho,warm_start_partition,100,20,1.0e-2,1.0e-3,1.0e-3,100,block);
        sorted_idx = Array{Int64}(n*(n-1));
        sorted_idx[1:Wlen] = sortperm(W[1:Wlen]);
        W[1:Wlen] = W[sorted_idx[1:Wlen]];
        lamb[1:Wlen] = lamb[sorted_idx[1:Wlen]];
        phi,xi = get_primal_solution_partial(Xmat,Y,rho,W[1:Wlen],lamb);
        obj = 0
        L = 0;
        if heuristics == true
            rule = -1
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(n);
            Wchecklen = Array{Int64,1}(n)
            for i in 1:n
                Wchecklen[i] = 0;
                Wcheck[i] = Array{Tuple{Int64,Int64},1}(J);
            end
        else
            Wcheck = Array{Array{Tuple{Int64,Int64},1},1}(0);
            Wchecklen = Array{Int64,1}(0)
        end
    end
    if LS == true
        L = 0
    end
    
    for k in 1:maxiter
        Delta = active_set_augmentation(Xmat,Y,W[1:Wlen],Wlen,phi,xi,lamb,violTOL,rule,P,M,K,G,block,J,Wcheck,Wchecklen);
        Delta_len = length(Delta);
        if verbose >= 1
            println("Iteration:")
            println(k);
        end
        if Delta_len != 0
            Wlen_new = Wlen + Delta_len
            W[Wlen+1:Wlen_new] = Delta
            sorted_idx[1:Wlen_new] = sortperm(W[1:Wlen_new]);
            W[1:Wlen_new] = W[sorted_idx[1:Wlen_new]];
            lamb[1:Wlen_new] = lamb[sorted_idx[1:Wlen_new]];
            flag = 0
            if verbose >= 1
                println(Wlen)
            end
            Wlen = Wlen_new
            if verbose >= 1
                println(Wlen)
            end
        else
            flag += 1
            if verbose >= 1
                println(Wlen)
                println(Wlen)
            end
        end
        tmp_steps = max_steps
        if inexact == true
            if  k == maxiter
                tmp_steps = 100
            else
                tmp_steps = max_steps
            end
        end
        if LS == true
            phi,xi,lamb,L,obj = cvxreg_qp_ineq_APG_ls(Xmat, Y, rho, W[1:Wlen], Wlen, lamb, tmp_steps, L,innerTOL)
        else
            phi,xi,lamb,obj = cvxreg_qp_ineq_APG(Xmat, Y, rho, W[1:Wlen], Wlen, lamb, max_steps, innerTOL)
        end
        if k > 1&& obj - last_obj > -outerTOL*0.1 && obj > last_obj * (1+outerTOL) && flag >= 5
            if verbose >= 1
                println(obj)
            end
            break
        else
            if verbose >= 1
                println(obj)
            end
            last_obj = obj;
        end
    end
    solvetime = toq();
    if verbose >= 1
        println(solvetime)
    end
    return phi,xi,lamb, W,Wlen,obj
end
