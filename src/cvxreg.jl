include("toolbox.jl")
include("solver.jl")
include("augmentation.jl")

function warm_start(Xmat::Array{T,2}, Y::Array{T,1}, rho::T, n::Int64, d::Int64; n_partition=0, partition_size=0,max_steps=5, maxiter=50, violTOL=1.0e-3, innerTOL=1.0e-4,outerTOL=1.0e-5,block=0) where T<:AbstractFloat
    Is = Array{Int64,1}(undef,0)
    Js = Array{Int64,1}(undef,0)
    prtt_len::Int64 = 0;
    obj::Float64 = 0.;
    lamb_vec = Array{eltype(Y),1}(undef, 0);
    if partition_size != 0 && n_partition != 0
        @assert partition_size*n_partition == n
        m = partition_size
    elseif partition_size !=0
        m = partition_size
    elseif n_partition != 0
        m = cld(n, n_partition);
    else
        m = cld(n, 5);
    end
    partition = partition_n_by_m(n,m);
    for prtt in partition
        prtt_len = length(prtt);
        _,_,lamb_sub, _, I_sub, J_sub, _ ,obj_sub = active_set(Xmat[prtt,:], Y[prtt], rho; max_steps=max_steps,maxiter=maxiter,violTOL=violTOL, 
            innerTOL=innerTOL, outerTOL=outerTOL, verbose=0,LBFGS=true, WS=false, block=block, aug = ActiveSetAugmentation(2,prtt_len,block,violTOL))
        append!(Is, prtt[I_sub[lamb_sub.nzval .!=0]]);
        append!(Js, prtt[J_sub[lamb_sub.nzval .!=0]]);
        append!(lamb_vec, lamb_sub[lamb_sub.!=0]);
        obj += obj_sub;
    end
    
    Wlen = length(lamb_vec);
    if block == 0
        W = sparse(Js, Is, fill(true, Wlen), n,n);
        lamb = sparse(Js, Is, lamb_vec, n,n);
    else
        W = sparse(Is, Js, fill(true, Wlen), n,n);
        lamb = sparse(Is, Js, lamb_vec, n,n);
    end
    return W, lamb, Wlen, obj, Is, Js;
end


function active_set(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; 
        max_steps= 5,maxiter=50,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, 
        verbose = 1,random_state = 42,LBFGS=false, LS =true, WS=false, inexact = true, 
        block = 0, aug = ActiveSetAugmentation(2,n,block,violTOL)) where {T<:AbstractFloat}
    start = time()
    if random_state != -1
        Random.seed!(random_state);
    end
    n,d = size(Xmat);
    Xflat = mat_to_flat(Xmat,n,d);
    last_obj = 0.;
    flag::Int = 0;
    if ~WS
        W = spzeros(Bool, n,n);
        Wlen::Int64 = 0;
        Wlen_prev::Int64 = 0;
        lamb = spzeros(eltype(Y),n,n);
        phi = copy(Y);
        xi = zeros(eltype(Y),n*d);
        obj = 0.;
        if block ==0
            J,I,_ = findnz(W)
        else
            I,J,_ = findnz(W)
        end
    else
        W, lamb, Wlen, obj, I, J = warm_start(Xmat, Y, rho, n, d; n_partition=5);
        Wlen_prev = Wlen;
        phi = zeros(eltype(Y), n);
        xi = zeros(eltype(Y),n*d);
        get_primal_solution!(phi,xi,Xflat,Y,rho,I,J, Wlen, lamb.nzval, n, d)
    end
    if (~LBFGS)&&LS 
        L::Float64 = 0.;
    end
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if verbose >= 1
            println("\nIteration:  ",k)
            println("Wlen_prev:  ",Wlen_prev)
        end
        Wlen = aug(phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
        if Wlen > Wlen_prev
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        end
        if Wlen - Wlen_prev <= 0.005*n
            flag += 1
        else
            flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
            println("     flag:  ",flag)
        end 
        if LBFGS == false
            if LS == true
                L,obj = cvxreg_qp_ineq_APG_ls(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
                println("        L:  ",L)
            else
                obj = cvxreg_qp_ineq_APG(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
            end
        else
            obj = cvxreg_qp_ineq_LBFGS(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
        end
        if verbose >= 1
            println("Objective:  ",obj)
        end
        if k > 1&& (obj - last_obj > -outerTOL || obj > last_obj * (1+outerTOL)) && flag >= 5
            break
        else
            last_obj = obj;
        end
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime," s.")
    end
    return phi,xi,lamb, W, I,J, Wlen,obj
end


function active_set_profiling(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; 
        max_steps= 5,maxiter=50,maxtime=10800,violTOL =1.0e-3, innerTOL=1.0e-4, outerTOL=1.0e-5, 
        verbose = 1,random_state = 42,LBFGS=false, LS =true, WS=false, inexact = true, 
        block = 0, aug = ActiveSetAugmentation(2,n,block,violTOL)) where {T<:AbstractFloat}
    start = time()
    cur_time = @elapsed begin
        if random_state != -1
            Random.seed!(random_state);
        end
        n,d = size(Xmat);
        Xflat =  mat_to_flat(Xmat,n,d);
        last_obj = 0.;
        flag::Int = 0;
        if ~WS
            W = spzeros(Bool, n,n);
            Wlen::Int64 = 0;
            Wlen_prev::Int64 = 0;
            lamb = spzeros(eltype(Y),n,n);
            phi = copy(Y);
            xi = zeros(eltype(Y),n*d);
            obj = 0.;
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        else
            W, lamb, Wlen, obj, I, J = warm_start(Xmat, Y, rho, n, d; n_partition=5);
            Wlen_prev = Wlen;
            phi = zeros(eltype(Y), n);
            xi = zeros(eltype(Y),n*d);
            get_primal_solution!(phi,xi,Xflat,Y,rho,I,J, Wlen, lamb.nzval, n, d)
        end
        if (~LBFGS)&&LS 
            L::Float64 = 0.;
        end
    end
    searching_times = Array{Float64}([]) # len = # outer 
    all_objs = Array{Float64}([]) # len = # inner + outer
    all_obj_times = Array{Float64}([]) # len = # inner + outer
    Wlen_list = Array{Int64}([]) # len = # inner + 1
    if LBFGS
        pow_times = nothing
        L_list = nothing
        f_evals_list = Array{Int64}([]) # len = # outer
    elseif ~LS
        pow_times = Array{Float64}([]) # len = # outer
        f_evals_list = nothing
        L_list = Array{Float64}([]) # len = # outer
    elseif LS
        pow_times = nothing
        L_list = Array{Float64}([]) # len = # outer
        f_evals_list = Array{Int64}([]) # len = # outer
    end

    optimization_starting_steps = Array{Int64}([]) # len = # outer 
    searching_time ::Float64 = 0;
    append!(all_objs, last_obj)
    append!(all_obj_times,cur_time)
    append!(Wlen_list, Wlen)
    solvetime::Float64 = 0;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if k >= 2 && searching_times[end]+cur_time >= maxtime
            println("early stopping in searching due to time limit")
            break
        end
        if verbose >= 1
            println("\nIteration:  ", k)
            println("Wlen_prev:  ",Wlen_prev)
        end
        searching_time = @elapsed Wlen = aug(phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
        cur_time += searching_time
        append!(searching_times,searching_time)
        append!(Wlen_list, Wlen)
        if Wlen > Wlen_prev
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        end
        if Wlen - Wlen_prev <= 0.005*n
            flag += 1
        else
            flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
            println("     flag:  ",flag)
        end 
        append!(optimization_starting_steps, length(all_objs)+1)
        if LBFGS == false
            if LS == true
                solvetime = @elapsed status, L, obj, times, objs, f_evals, Ls, _ = cvxreg_qp_ineq_APG_ls_profiling(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps, 
                    start_time = cur_time, end_time = maxtime)
                append!(f_evals_list, sum(f_evals));
                append!(L_list, L);
                println("        L:  ",L)
            else
                solvetime = @elapsed status, obj, times, objs, pow_time, L, _ = cvxreg_qp_ineq_APG_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps,
                    start_time = cur_time, end_time = maxtime)
                append!(pow_times, pow_time)
                append!(L_list, L)
                println("        L:  ",L)
            end
        else
            solvetime = @elapsed obj, times, objs, f_evals, _ = cvxreg_qp_ineq_LBFGS_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
            append!(f_evals_list, sum(f_evals)) 
            status= 0
        end
        append!(all_objs,objs)
        append!(all_obj_times,cur_time.+times)
        cur_time += solvetime;
        if verbose >= 1
            println("Objective:  ",obj)
            println("     Time:  ",all_obj_times[end])
        end
        if status == -1
            break
        end
        if cur_time >= maxtime
            println("early stopping after optimization due to time limit")
            break
        end
        if k > 1&& (obj - last_obj > -outerTOL || obj > last_obj * (1+outerTOL)) && flag >= 5
            break
        else
            last_obj = obj;
        end
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime," s.")
    end
    return phi,xi,lamb, W, Wlen, all_objs,all_obj_times,Wlen_list,searching_times,pow_times,f_evals_list, L_list, optimization_starting_steps
end


struct AlgorithmParameters
    LBFGS::Bool;
    LS::Bool;
    WS::Bool;
    inexact::Bool;
    max_steps::Int64;
    violTOL::Float64;
    innerTOL::Float64;
    outerTOL::Float64;
    function AlgorithmParameters(;LBFGS=false,LS=true,WS=false,inexact=true,max_steps=-1,violTOL=1.0e-3,innerTOL=-1.,outerTOL=1.0e-5)
        if inexact
            max_steps = (max_steps == -1) ? 5 : max_steps
            innerTOL = (innerTOL==-1.0) ? 1e-6 : innerTOL
        else
            max_steps = (max_steps == -1) ? 3000 : max_steps
            innerTOL = (innerTOL==-1.0) ? 1e-7 : innerTOL
        end
        new(LBFGS, LS, WS, inexact, max_steps, violTOL, innerTOL, outerTOL);
    end
end


function two_stage_active_set(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; verbose = 1, random_state = 42, maxiter = 50, block = 0, dropzero = true,
        params_list::Array{AlgorithmParameters,1}, augs::Array{ActiveSetAugmentation,1}) where {T<:AbstractFloat}
    start = time()
    if random_state != -1
        Random.seed!(random_state);
    end
    n,d = size(Xmat);
    Xflat = mat_to_flat(Xmat,n,d);
    last_obj = 0.;
    flag::Int = 0;
    WS = params_list[1].WS
    LS = params_list[1].LS
    LBFGS = params_list[1].LBFGS
    if ~WS
        W = spzeros(Bool, n,n);
        Wlen::Int64 = 0;
        Wlen_prev::Int64 = 0;
        lamb = spzeros(eltype(Y),n,n);
        phi = copy(Y);
        xi = zeros(eltype(Y),n*d);
        obj = 0.;
        if block ==0
            J,I,_ = findnz(W)
        else
            I,J,_ = findnz(W)
        end
    else
        W, lamb, Wlen, obj, I, J = warm_start(Xmat, Y, rho, n, d; n_partition=5);
        Wlen_prev = Wlen;
        phi = zeros(eltype(Y), n);
        xi = zeros(eltype(Y),n*d);
        get_primal_solution!(phi,xi,Xflat,Y,rho,I,J, Wlen, lamb.nzval, n, d)
    end
    if (~LBFGS)&&LS 
        L::Float64 = 0.;
    end
    stage::Int64 = 1;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if verbose >= 1
            println("\nIteration:  ", k)
            println("    Stage:  ", stage)
            println("Wlen_prev:  ",Wlen_prev)
        end
        Wlen = augs[stage](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
        if Wlen > Wlen_prev
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        end
        if Wlen - Wlen_prev <= 0.005*n
            flag += 1
        elseif stage == 1
            flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
            println("     flag:  ",flag)
        end 
        if LBFGS == false
            if LS == true
                L,obj = cvxreg_qp_ineq_APG_ls(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
                println("        L:  ",L)
            else
                obj = cvxreg_qp_ineq_APG(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
            end
        else
            obj = cvxreg_qp_ineq_LBFGS(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
        end
        if verbose >= 1
            println("Objective:  ",obj)
        end
        if stage == 1 && k > 1 && flag >= 5
            flag = 0;
            stage = 2;
            if dropzero
                lamb = dropzeros(lamb)
                if block ==0
                    J,I,_ = findnz(lamb)
                else
                    I,J,_ = findnz(lamb)
                end
                Wlen = length(lamb.nzval)
                Wlen_prev = Wlen
                W = sparse(J,I,fill(true,Wlen),n,n)
            end
        elseif stage == 2 && (k > 1&& (obj - last_obj > -params_list[stage].outerTOL || obj > last_obj * (1+params_list[stage].outerTOL)))
            if flag >= 10
                break
            else
                flag += 1
            end
        end
        last_obj = obj;
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime," s.")
    end
    return phi,xi,lamb, W, I,J, Wlen,obj
end


function two_stage_active_set_profiling(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; verbose = 1, random_state = 42, maxiter = 50, maxtime = 10800, block = 0, dropzero = true,
        params_list::Array{AlgorithmParameters,1}, augs::Array{ActiveSetAugmentation,1}) where {T<:AbstractFloat}
    start = time()
    cur_time = @elapsed begin
        if random_state != -1
            Random.seed!(random_state);
        end
        n,d = size(Xmat);
        Xflat = mat_to_flat(Xmat,n,d);
        last_obj = 0.;
        flag::Int = 0;
        WS = params_list[1].WS
        LS = params_list[1].LS
        LBFGS = params_list[1].LBFGS
        if ~WS
            W = spzeros(Bool, n,n);
            Wlen::Int64 = 0;
            Wlen_prev::Int64 = 0;
            lamb = spzeros(eltype(Y),n,n);
            phi = copy(Y);
            xi = zeros(eltype(Y),n*d);
            obj = 0.;
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        else
            W, lamb, Wlen, obj, I, J = warm_start(Xmat, Y, rho, n, d; n_partition=5);
            Wlen_prev = Wlen;
            phi = zeros(eltype(Y), n);
            xi = zeros(eltype(Y),n*d);
            get_primal_solution!(phi,xi,Xflat,Y,rho,I,J, Wlen, lamb.nzval, n, d)
        end
        if (~LBFGS)&&LS 
            L::Float64 = 0.;
        end
        stage::Int64 = 1;
    end
    stages = Array{Int64}([]) # len = # outer
    searching_times = Array{Float64}([]) # len = # outer 
    all_objs = Array{Float64}([]) # len = # inner + outer
    all_obj_times = Array{Float64}([]) # len = # inner + outer
    Wlen_list = Array{Int64}([]) # len = # inner + 1
    if LBFGS
        pow_times = nothing
        L_list = nothing
        f_evals_list = Array{Int64}([]) # len = # outer
    elseif ~LS
        pow_times = Array{Float64}([]) # len = # outer
        f_evals_list = nothing
        L_list = Array{Float64}([]) # len = # outer
    elseif LS
        pow_times = nothing
        L_list = Array{Float64}([]) # len = # outer
        f_evals_list = Array{Int64}([]) # len = # outer
    end

    optimization_starting_steps = Array{Int64}([]) # len = # outer 
    searching_time ::Float64 = 0;
    append!(all_objs, last_obj)
    append!(all_obj_times,cur_time)
    append!(Wlen_list, Wlen)
    solvetime::Float64 = 0;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if k >= 2 && searching_times[end]+cur_time >= maxtime
            println("early stopping in searching due to time limit")
            break
        end
        append!(stages, stage)
        if verbose >= 1
            println("\nIteration:  ", k)
            println("    Stage:  ", stage)
            println("Wlen_prev:  ",Wlen_prev)
        end
        searching_time = @elapsed Wlen = augs[stage](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
        cur_time += searching_time
        append!(searching_times,searching_time)
        append!(Wlen_list, Wlen)
        if Wlen > Wlen_prev
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        end
        if Wlen - Wlen_prev <= 0.005*n
            flag += 1
        elseif stage == 1
            flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
            println("     flag:  ",flag)
        end 
        append!(optimization_starting_steps, length(all_objs)+1)
        if LBFGS == false
            if LS == true
                solvetime = @elapsed status, L, obj, times, objs, f_evals, Ls, _ = cvxreg_qp_ineq_APG_ls_profiling(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps, start_time = cur_time, end_time = maxtime)
                append!(f_evals_list, sum(f_evals));
                append!(L_list, L);
                println("        L:  ",L)
            else
                solvetime = @elapsed status, obj, times, objs, pow_time, L, _ = cvxreg_qp_ineq_APG_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps,  start_time = cur_time, end_time = maxtime)
                append!(pow_times, pow_time)
                append!(L_list, L)
                println("        L:  ",L)
            end
        else
            solvetime = @elapsed obj, times, objs, f_evals, _ = cvxreg_qp_ineq_LBFGS_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
            append!(f_evals_list, sum(f_evals))
            status = 0;
        end
        append!(all_objs,objs)
        append!(all_obj_times,cur_time.+times)
        cur_time += solvetime;
        if verbose >= 1
            println("Objective:  ",obj)
            println("     Time:  ",all_obj_times[end])
        end
        if status == -1
            break
        end
        if cur_time >= maxtime
            println("early stopping after optimization due to time limit")
            break
        end
        if stage == 1 && k > 1 && flag >= 5
            flag = 0;
            stage = 2;
            if dropzero
                lamb = dropzeros(lamb)
                if block ==0
                    J,I,_ = findnz(lamb)
                else
                    I,J,_ = findnz(lamb)
                end
                Wlen = length(lamb.nzval)
                Wlen_prev = Wlen
                W = sparse(J,I,fill(true,Wlen),n,n)
            end
        elseif stage == 2 &&  (k > 1&& (obj - last_obj > -params_list[stage].outerTOL || obj > last_obj * (1+params_list[stage].outerTOL)))
            if flag >= 10
                break
            else
                flag += 1
            end
        end
        last_obj = obj;
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime," s.")
    end
    return phi,xi,lamb, W, Wlen, all_objs,all_obj_times, stages,Wlen_list,searching_times,pow_times,f_evals_list, L_list, optimization_starting_steps
end


