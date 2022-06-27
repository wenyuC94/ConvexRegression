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
        L_init_estimate = :power, decrease_L = false,linesearch_type=:sufficient_decrease,
        block = 0, aug = ActiveSetAugmentation(2,n,block,violTOL)) where {T<:AbstractFloat}
    start = time()
    if random_state != -1
        Random.seed!(random_state);
    end
    n,d = size(Xmat);
    Xflat = reshape(copy(Xmat'),(n*d));
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
            flag = 0
            Wlen_prev = Wlen
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        else
            flag += 1
        end
        if verbose >= 1
            println("     Wlen:  ",Wlen)
        end 
        if LBFGS == false
            if LS == true
                L,obj = cvxreg_qp_ineq_APG_ls(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps, L_init_estimate = L_init_estimate, decrease_L = decrease_L, linesearch_type=linesearch_type)
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
        if k > 1&& obj - last_obj > -outerTOL && obj > last_obj * (1+outerTOL) && flag >= 5
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
        L_init_estimate = :power, decrease_L = false, linesearch_type=:sufficient_decrease,
        block = 0, aug = ActiveSetAugmentation(2,n,block,violTOL), evaluation=:none, evaluation_freq=0::Int64, 
        gap_evaluation = true, randomized_evaluation=false, randomized_ratio = 10) where {T<:AbstractFloat}
    start = time()
    cur_time = @elapsed begin
        if random_state != -1
            Random.seed!(random_state);
        end
        n,d = size(Xmat);
        Xflat = reshape(copy(Xmat'),(n*d));
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
            last_obj = obj;
        end
        if (~LBFGS)&&LS 
            L::Float64 = 0.;
        end
    end
    
    ## inner profile
    all_objs = Array{Float64}([last_obj]) # len = # inner + outer
    all_obj_times = Array{Float64}([cur_time]) # len = # inner + outer
    
    ## outer profile
    Wlen_list = Array{Int64}([Wlen]) # len = # outer + 1
    stages = Array{Int64}([1]) # len = # outer + 1
    searching_times = Array{Union{Float64,Missing}}([missing]) # len = # outer + 1
    optimization_starting_steps = Array{Int64}([-1]) # len = # outer + 1
    if LBFGS
        pow_times = nothing
        L_list = nothing
        f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
    elseif ~LS
        pow_times = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
        f_evals_list = nothing
        L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
    elseif LS
        pow_times = nothing
        L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
        f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
    end
    
    searching_time ::Float64 = 0;
    
    
    ## evaluation list
    pinfeas_list = Array{Float64}([])
    maxv_list = Array{Float64}([])
    eval_time_list = Array{Float64}([])
    eval_inner_list = Array{Int64}([])
    eval_outer_list = Array{Int64}([])
    eval_elapsed_list = Array{Float64}([])
    dgap_list = Array{Float64}([])
    bgap_list = Array{Float64}([])
    ub_list = Array{Float64}([])
    lb_list = Array{Float64}([])
    cur_eval_thresh::Int64 = evaluation_freq
    cur_eval::Float64 = 0.
    eval_flag = false
    eval_elapsed::Float64 = 0.
    pinfeas::Float64 = 0.
    max_viol::Float64 = 0.
    obj_ub::Float64 = 0.
    obj_lb::Float64 = 0.
    dual_gap::Float64 = 0.
    best_gap::Float64 = 0.
    
    solvetime::Float64 = 0;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if verbose >= 1
            println("\nIteration:  ", k)
            println("Wlen_prev:  ",Wlen_prev)
        end
        searching_time = @elapsed Wlen = aug(phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
        cur_time += searching_time
        append!(searching_times,searching_time)
        append!(Wlen_list, Wlen)
        if Wlen > Wlen_prev
            flag = 0
            Wlen_prev = Wlen
            if block ==0
                J,I,_ = findnz(W)
            else
                I,J,_ = findnz(W)
            end
        else
            flag += 1
        end
        if verbose >= 1
            println("     Wlen:  ",Wlen)
        end 
        append!(optimization_starting_steps, length(all_objs)+1)
        if LBFGS == false
            if LS == true
                solvetime = @elapsed L, obj, times, objs, f_evals, Ls, _ = cvxreg_qp_ineq_APG_ls_profiling(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps, L_init_estimate = L_init_estimate, decrease_L = decrease_L, linesearch_type=linesearch_type)
                append!(f_evals_list, sum(f_evals));
                append!(L_list, L);
                println("        L:  ",L)
            else
                solvetime = @elapsed obj, times, objs, pow_time, L, _ = cvxreg_qp_ineq_APG_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
                append!(pow_times, pow_time)
                append!(L_list, L)
                println("        L:  ",L)
            end
        else
            solvetime = @elapsed obj, times, objs, f_evals, _ = cvxreg_qp_ineq_LBFGS_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; TOL=innerTOL, max_steps = max_steps)
            append!(f_evals_list, sum(f_evals))            
        end
        append!(all_objs,objs)
        append!(all_obj_times,cur_time.+times)
        cur_time += solvetime;
        if verbose >= 1
            println("Objective:  ",obj)
            println("     Time:  ",all_obj_times[end])
            flush(stdout)
        end
        if cur_time >= maxtime
            break
        end
        
        if evaluation == :time
            cur_eval = cur_time
        elseif evaluation == :inner
            cur_eval = length(all_objs)
        elseif evaluation == :outer
            cur_eval = length(searching_times)
        elseif evaluation == :none
            cur_eval = 0
        end
        if (evaluation != :none) && (cur_eval >= cur_eval_thresh)
            eval_flag = true
            while (cur_eval >= cur_eval_thresh)
                cur_eval_thresh += evaluation_freq
            end
        end
        if eval_flag
            if verbose >= 1
                println("\nEvaluation started...")
            end
            eval_elapsed = @elapsed dual_gap, pinfeas, max_viol, obj_ub, obj_lb = get_dual_solution_quality(Xflat,Y,rho,I,J, Wlen, lamb.nzval, phi, xi, n, d; 
                return_gap=gap_evaluation, randomized=randomized_evaluation, div=randomized_ratio)
            append!(pinfeas_list, pinfeas)
            append!(maxv_list, max_viol)
            append!(dgap_list, dual_gap)
            append!(lb_list, obj_lb)
            append!(ub_list, obj_ub)
            append!(eval_time_list, cur_time)
            append!(eval_inner_list, length(all_objs))
            append!(eval_outer_list, length(searching_times))
            append!(eval_elapsed_list, eval_elapsed)
            best_gap = obj_ub - maximum(lb_list)
            append!(bgap_list, best_gap)
            if verbose >= 1
                println("Primal Infeasibility: ", pinfeas)
                println("Max Violation       : ", max_viol)
                println("Upper Bound         : ", obj_ub)
                println("Lower Bound         : ", obj_lb)
                println("Current Duality Gap : ", dual_gap)
                println("Best Duality Gap    : ", best_gap)
                println("Evaluation Runtime  : ", eval_elapsed)
                flush(stdout)
            end
            
            eval_flag = false
        end
        
        
        
        if k > 1&& obj - last_obj > -outerTOL && obj > last_obj * (1+outerTOL) && flag >= 5
            break
        else
            last_obj = obj;
        end
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime-sum(eval_elapsed_list)," s.")
    end
    
    inner_profile_dict = OrderedDict{String, Any}(
        "step"=>Array(0:length(all_objs)-1),
        "objs"=>all_objs,
        "time"=>all_obj_times
    )
    
    outer_profile_dict = OrderedDict{String, Any}(
        "iter"=>Array(0:length(Wlen_list)-1),
        "augmentation"=>searching_times,
        "power"=> (pow_times==nothing) ? fill(missing, length(Wlen_list)) : pow_times,
        "f_eval"=> (f_evals_list==nothing) ? fill(missing, length(Wlen_list)) : f_evals_list,
        "L"=> (L_list==nothing) ? fill(missing, length(Wlen_list)) : L_list,
        "Wlen"=>Wlen_list,
        "starting_steps"=>optimization_starting_steps
    )
    
    eval_KKT_dict = OrderedDict{String,Any}(
            "pinfeas"=>pinfeas_list, 
            "max_viol"=>maxv_list,
            "obj_ub"=>ub_list, 
            "obj_lb"=>lb_list, 
            "dual_gap"=>dgap_list, 
            "best_gap"=>bgap_list, 
            "eval_time"=>eval_time_list, 
            "eval_inner"=>eval_inner_list, 
            "eval_outer"=>eval_outer_list, 
            "elapsed"=>eval_elapsed_list
        )
    
    return phi,xi,lamb, W, Wlen, inner_profile_dict, outer_profile_dict, eval_KKT_dict
end


struct AlgorithmParameters
    LBFGS::Bool;
    LS::Bool;
    WS::Bool;
    inexact::Bool;
    max_steps::Int64;
    min_steps::Int64;
    L_init_estimate::Symbol;
    decrease_L::Bool;
    linesearch_type::Symbol;
    violTOL::Float64;
    innerTOL::Float64;
    outerTOL::Float64;
    function AlgorithmParameters(;LBFGS=false,LS=true,WS=false,inexact=true,max_steps=-1,L_init_estimate=:twopoints,decrease_L=false,linesearch_type=:sufficient_decrease, violTOL=1.0e-3,innerTOL=-1.,outerTOL=1.0e-5)
        if inexact
            max_steps = (max_steps == -1) ? 5 : max_steps
            min_steps = (max_steps == -1) ? 0 : 0
            innerTOL = (innerTOL==-1.0) ? 1e-6 : innerTOL
        else
            max_steps = (max_steps == -1) ? 3000 : max_steps
            min_steps = min(5, max_steps)
            innerTOL = (innerTOL==-1.0) ? 1e-7 : innerTOL
        end
        new(LBFGS, LS, WS, inexact, max_steps,min_steps,L_init_estimate, decrease_L, linesearch_type, violTOL, innerTOL, outerTOL);
    end
end


function two_stage_active_set_profiling(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; verbose = 1, random_state = 42, maxiter = 50, maxtime = 10800, block = 0, reduction_while_switch = true,
        greedy_while_switch = false, params_list::Array{AlgorithmParameters,1}, augs::Array{ActiveSetAugmentation,1}, 
        evaluation=:none, evaluation_freq=0::Int64,gap_evaluation = true, randomized_evaluation=false, randomized_ratio = 10, 
        reduction_then_greedy = false, greedy_augs::Array{ActiveSetAugmentation,1}) where {T<:AbstractFloat}
    @assert length(augs) == 2
    if reduction_then_greedy || greedy_while_switch
        @assert length(greedy_augs) >= 1
    end
    start = time()
    cur_time = @elapsed begin
        if random_state != -1
            Random.seed!(random_state);
        end
        n,d = size(Xmat);
        Xflat = reshape(copy(Xmat'),(n*d));
        last_obj = 0.;
        Wlen_flag::Int = 0;
        feval_flag::Int = 0;
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
            last_obj = obj;
        end
        if (~LBFGS)&&LS 
            L::Float64 = 0.;
        end
        stage::Int64 = 1;
    end
    
    ## inner profile
    all_objs = Array{Float64}([last_obj]) # len = # inner + outer
    all_obj_times = Array{Float64}([cur_time]) # len = # inner + outer
    
    ## outer profile
    Wlen_list = Array{Int64}([Wlen]) # len = # outer + 1
    stages = Array{Int64}([1]) # len = # outer + 1
    searching_times = Array{Union{Float64,Missing}}([missing]) # len = # outer + 1
    optimization_starting_steps = Array{Int64}([-1]) # len = # outer + 1
    if LBFGS
        pow_times = nothing
        L_list = nothing
        f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
    elseif ~LS
        pow_times = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
        f_evals_list = nothing
        L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
    elseif LS
        pow_times = nothing
        L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
        f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
    end
    
    searching_time ::Float64 = 0;
    ## rule1 profile
#     if reduction_then_greedy 
    num_rule1::Int64 = 0
    extra_searching_times = Array{Float64}([]) # len = num_rule1
    extra_searching_time ::Float64 = 0;
    Wlens_before_dropping = Array{Int64}([]) # len = num_rule1
    Wlens_after_dropping = Array{Int64}([]) # len = num_rule1
    Wlens_after_greedy = Array{Int64}([]) # len = num_rule1
    greedy_rule_outer = Array{Int64}([]) # len = num_rule1
#     end
    
    
    ## evaluation list
    pinfeas_list = Array{Float64}([])
    maxv_list = Array{Float64}([])
    eval_time_list = Array{Float64}([])
    eval_inner_list = Array{Int64}([])
    eval_outer_list = Array{Int64}([])
    eval_elapsed_list = Array{Float64}([])
    dgap_list = Array{Float64}([])
    bgap_list = Array{Float64}([])
    ub_list = Array{Float64}([])
    lb_list = Array{Float64}([])
    cur_eval_thresh::Int64 = evaluation_freq
    cur_eval::Float64 = 0.
    eval_flag = false
    eval_elapsed::Float64 = 0.
    pinfeas::Float64 = 0.
    max_viol::Float64 = 0.
    obj_ub::Float64 = 0.
    obj_lb::Float64 = 0.
    dual_gap::Float64 = 0.
    best_gap::Float64 = 0.

    solvetime::Float64 = 0;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        append!(stages, stage)
        if verbose >= 1
            println("\nIteration:  ", k)
            println("    Stage:  ", stage)
            if reduction_then_greedy
                println("   Greedy:  ", num_rule1)
            end
            println("Wlen_prev:  ",Wlen_prev)
        end
        searching_time =  @elapsed Wlen = augs[stage](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
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
        if Wlen - Wlen_prev <= 0.005*n ## for test
            Wlen_flag += 1
        else
            Wlen_flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
        end 
        append!(optimization_starting_steps, length(all_objs)+1)
        if LBFGS == false
            if LS == true
                solvetime = @elapsed L, obj, times, objs, f_evals, Ls, _ = cvxreg_qp_ineq_APG_ls_profiling(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps, min_steps = params_list[stage].min_steps, 
                    L_init_estimate = params_list[stage].L_init_estimate, decrease_L = params_list[stage].decrease_L, linesearch_type=params_list[stage].linesearch_type)
                append!(f_evals_list, sum(f_evals));
                append!(L_list, L);
                println("        L:  ",L)
                if length(objs) <= params_list[stage].min_steps+1
                    feval_flag += 1
                end
            else
                solvetime = @elapsed obj, times, objs, pow_time, L, _ = cvxreg_qp_ineq_APG_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps, min_steps = params_list[stage].min_steps)
                append!(pow_times, pow_time)
                append!(L_list, L)
                println("        L:  ",L)
            end
        else
            solvetime = @elapsed obj, times, objs, f_evals, _ = cvxreg_qp_ineq_LBFGS_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
            append!(f_evals_list, sum(f_evals))     
            if length(objs) <= params_list[stage].min_steps+1
                feval_flag += 1
            end
        end
        append!(all_objs,objs)
        append!(all_obj_times,cur_time.+times)
        cur_time += solvetime;
        if verbose >= 1
            println("Objective:  ",obj)
            println("     Time:  ",all_obj_times[end])
            flush(stdout)
        end
        if cur_time >= maxtime
            break
        end
        if evaluation == :time
            cur_eval = cur_time
        elseif evaluation == :inner
            cur_eval = length(all_objs)
        elseif evaluation == :outer
            cur_eval = length(searching_times)
        elseif evaluation == :none
            cur_eval = 0
        end
        if (evaluation != :none) && (cur_eval >= cur_eval_thresh)
            eval_flag = true
            while (cur_eval >= cur_eval_thresh)
                cur_eval_thresh += evaluation_freq
            end
        end
        if eval_flag
            if verbose >= 1
                println("\nEvaluation started...")
            end
            eval_elapsed = @elapsed dual_gap, pinfeas, max_viol, obj_ub, obj_lb = get_dual_solution_quality(Xflat,Y,rho,I,J, Wlen, lamb.nzval, phi, xi, n, d;
                return_gap=gap_evaluation, randomized=randomized_evaluation, div=randomized_ratio)
            append!(pinfeas_list, pinfeas)
            append!(maxv_list, max_viol)
            append!(dgap_list, dual_gap)
            append!(lb_list, obj_lb)
            append!(ub_list, obj_ub)
            append!(eval_time_list, cur_time)
            append!(eval_inner_list, length(all_objs))
            append!(eval_outer_list, length(searching_times))
            append!(eval_elapsed_list, eval_elapsed)
            best_gap = obj_ub - maximum(lb_list)
            append!(bgap_list, best_gap)
            if verbose >= 1
                println("Primal Infeasibility: ", pinfeas)
                println("Max Violation       : ", max_viol)
                println("Upper Bound         : ", obj_ub)
                println("Lower Bound         : ", obj_lb)
                println("Current Duality Gap : ", dual_gap)
                println("Best Duality Gap    : ", best_gap)
                println("Evaluation Runtime  : ", eval_elapsed)
                flush(stdout)
            end
            
            eval_flag = false
        end
        if stage == 1 && k > 1 && Wlen_flag >= 5 ## for test
            Wlen_flag = 0;
            feval_flag = 0;
            stage = 2;
            append!(greedy_rule_outer, length(all_objs))
            append!(Wlens_before_dropping, Wlen)
            if verbose >= 1
                println("Stage switched...")
                println("Wlen before dropping: ", Wlen)
            end
            if reduction_while_switch
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
            if verbose >= 1
                println("Wlen after dropping:  ", Wlen_prev)
            end
            if greedy_while_switch
                extra_searching_time = @elapsed begin
                    Wlen = greedy_augs[1](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
                    if Wlen > Wlen_prev
                        if block ==0
                            J,I,_ = findnz(W)
                        else
                            I,J,_ = findnz(W)
                        end
                    end
                end
                cur_time += extra_searching_time
            else
                extra_searching_time = 0
            end
            append!(extra_searching_times, extra_searching_time)
            append!(Wlens_after_dropping, Wlen_prev)
            append!(Wlens_after_greedy, Wlen)
            if verbose >= 1 && greedy_while_switch
                println("Wlen after Greedy:    ", Wlen)
            end
            
        elseif stage == 2 && (k > 1&& obj - last_obj > -params_list[stage].outerTOL && obj > last_obj * (1+params_list[stage].outerTOL) && Wlen_flag >= 5)
            break
        end
        
        if stage == 2
            if reduction_then_greedy && max(feval_flag, Wlen_flag) >= 5 ## for test
                num_rule1+=1
                append!(greedy_rule_outer, length(all_objs))
                append!(Wlens_before_dropping, Wlen)
                if verbose >= 1
                    println("dropzero with Greedy Rule started...")
                    println("Wlen before dropping: ", Wlen)
                end
                cur_time += @elapsed begin
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
                extra_searching_time = @elapsed begin
                    Wlen = greedy_augs[end](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
                    if Wlen > Wlen_prev
                        if block ==0
                            J,I,_ = findnz(W)
                        else
                            I,J,_ = findnz(W)
                        end
                    end
                end
                cur_time += extra_searching_time
                append!(extra_searching_times, extra_searching_time)
                append!(Wlens_after_dropping, Wlen_prev)
                append!(Wlens_after_greedy, Wlen)
                if verbose >= 1
                    println("Wlen after dropping:  ", Wlen_prev)
                    println("Wlen after Rule1:     ", Wlen)
                end
                feval_flag = 0;
                Wlen_flag = 0;
            end
        end
        Wlen_prev = Wlen;        
        last_obj = obj;
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime-sum(eval_elapsed_list)," s.")
    end
    
    inner_profile_dict = OrderedDict{String, Any}(
        "step"=>Array(0:length(all_objs)-1),
        "objs"=>all_objs,
        "time"=>all_obj_times
    )
    
    outer_profile_dict = OrderedDict{String, Any}(
        "iter"=>Array(0:length(Wlen_list)-1),
        "augmentation"=>searching_times,
        "power"=> (pow_times==nothing) ? fill(missing, length(Wlen_list)) : pow_times,
        "f_eval"=> (f_evals_list==nothing) ? fill(missing, length(Wlen_list)) : f_evals_list,
        "L"=> (L_list==nothing) ? fill(missing, length(Wlen_list)) : L_list,
        "Wlen"=>Wlen_list,
        "starting_steps"=>optimization_starting_steps,
        "stages"=>stages
    )
    
    
    greedy_profile_dict = OrderedDict{String,Any}(
        "num_rule1"=>Array(0:length(extra_searching_times)-1),
        "augmentation"=>extra_searching_times,
        "Wlen_before_dropping"=>Wlens_before_dropping,
        "Wlen_after_dropping"=>Wlens_after_dropping,
        "Wlen_after_rule1"=>Wlens_after_greedy,
        "happening_outer_iter"=>greedy_rule_outer
    )
    
    
    eval_KKT_dict = OrderedDict{String,Any}(
            "pinfeas"=>pinfeas_list, 
            "max_viol"=>maxv_list,
            "obj_ub"=>ub_list, 
            "obj_lb"=>lb_list, 
            "dual_gap"=>dgap_list, 
            "best_gap"=>bgap_list, 
            "eval_time"=>eval_time_list, 
            "eval_inner"=>eval_inner_list, 
            "eval_outer"=>eval_outer_list, 
            "elapsed"=>eval_elapsed_list
        )
    
    return phi,xi,lamb, W, Wlen, inner_profile_dict, outer_profile_dict, eval_KKT_dict, greedy_profile_dict
end


function two_stage_active_set(Xmat::Array{T,2}, Y::Array{T,1}, rho::T; verbose = 1, random_state = 42, maxiter = 50, block = 0, reduction_while_switch = true,
        greedy_while_switch = false, params_list::Array{AlgorithmParameters,1}, augs::Array{ActiveSetAugmentation,1}) where {T<:AbstractFloat}
    start = time()
    @assert length(augs) == 2
    if reduction_then_greedy || greedy_while_switch
        @assert length(greedy_augs) >= 1
    end
    if random_state != -1
        Random.seed!(random_state);
    end
    n,d = size(Xmat);
    Xflat = reshape(copy(Xmat'),(n*d));
    last_obj = 0.;
    Wlen_flag::Int = 0;
    feval_flag::Int = 0;
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
        last_obj = obj;
    end
    if (~LBFGS)&&LS 
        L::Float64 = 0.;
    end
    stage::Int64 = 1;
    
    ## inner profile
#     all_objs = Array{Float64}([last_obj]) # len = # inner + outer
#     all_obj_times = Array{Float64}([cur_time]) # len = # inner + outer
    
    ## outer profile
#     Wlen_list = Array{Int64}([Wlen]) # len = # outer + 1
#     stages = Array{Int64}([1]) # len = # outer + 1
#     searching_times = Array{Union{Float64,Missing}}([missing]) # len = # outer + 1
#     optimization_starting_steps = Array{Int64}([-1]) # len = # outer + 1
#     if LBFGS
#         pow_times = nothing
#         L_list = nothing
#         f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
#     elseif ~LS
#         pow_times = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
#         f_evals_list = nothing
#         L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
#     elseif LS
#         pow_times = nothing
#         L_list = Array{Union{Missing,Float64}}([missing]) # len = # outer+1
#         f_evals_list = Array{Union{Missing,Int64}}([missing]) # len = # outer+1
#     end
    
#     searching_time ::Float64 = 0;
#     ## rule1 profile
# #     if reduction_then_greedy 
    num_rule1::Int64 = 0
#     extra_searching_times = Array{Float64}([]) # len = num_rule1
#     extra_searching_time ::Float64 = 0;
#     Wlens_before_dropping = Array{Int64}([]) # len = num_rule1
#     Wlens_after_dropping = Array{Int64}([]) # len = num_rule1
#     Wlens_after_greedy = Array{Int64}([]) # len = num_rule1
#     greedy_rule_outer = Array{Int64}([]) # len = num_rule1
#     end
    
    
    ## evaluation list
#     pinfeas_list = Array{Float64}([])
#     maxv_list = Array{Float64}([])
#     eval_time_list = Array{Float64}([])
#     eval_inner_list = Array{Int64}([])
#     eval_outer_list = Array{Int64}([])
#     eval_elapsed_list = Array{Float64}([])
#     dgap_list = Array{Float64}([])
#     bgap_list = Array{Float64}([])
#     ub_list = Array{Float64}([])
#     lb_list = Array{Float64}([])
#     cur_eval_thresh::Int64 = evaluation_freq
#     cur_eval::Float64 = 0.
#     eval_flag = false
#     eval_elapsed::Float64 = 0.
#     pinfeas::Float64 = 0.
#     max_viol::Float64 = 0.
#     obj_ub::Float64 = 0.
#     obj_lb::Float64 = 0.
#     dual_gap::Float64 = 0.
#     best_gap::Float64 = 0.

#     solvetime::Float64 = 0;
    if verbose >= 1
        println("Optimization Started\n")
    end
    for k in 1:maxiter
        if verbose >= 1
            println("\nIteration:  ", k)
            println("    Stage:  ", stage)
            if reduction_then_greedy
                println("   Greedy:  ", num_rule1)
            end
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
        if Wlen - Wlen_prev <= 0.005*n ## for test
            Wlen_flag += 1
        else
            Wlen_flag = 0
        end
        Wlen_prev = Wlen;
        if verbose >= 1
            println("     Wlen:  ",Wlen)
        end 
        if LBFGS == false
            if LS == true
                L, obj, times, objs, f_evals, Ls, _ = cvxreg_qp_ineq_APG_ls_profiling(Xflat, Y, rho, I, J, Wlen, lamb, L, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps, min_steps = params_list[stage].min_steps, 
                    L_init_estimate = params_list[stage].L_init_estimate, decrease_L = params_list[stage].decrease_L, linesearch_type=params_list[stage].linesearch_type)
                println("        L:  ",L)
                if length(objs) <= params_list[stage].min_steps+1
                    feval_flag += 1
                end
            else
                obj, times, objs, pow_time, L, _ = cvxreg_qp_ineq_APG_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                    TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps, min_steps = params_list[stage].min_steps)
                println("        L:  ",L)
            end
        else
            @elapsed obj, times, objs, f_evals, _ = cvxreg_qp_ineq_LBFGS_profiling(Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi, xi; 
                TOL=params_list[stage].innerTOL, max_steps = params_list[stage].max_steps)
            append!(f_evals_list, sum(f_evals))     
            if length(objs) <= params_list[stage].min_steps+1
                feval_flag += 1
            end
        end
        if verbose >= 1
            println("Objective:  ",obj)
            flush(stdout)
        end
        if stage == 1 && k > 1 && Wlen_flag >= 5 ## for test
            Wlen_flag = 0;
            feval_flag = 0;
            stage = 2;
            if verbose >= 1
                println("Stage switched...")
                println("Wlen before dropping: ", Wlen)
            end
            if reduction_while_switch
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
            if verbose >= 1
                println("Wlen after dropping:  ", Wlen_prev)
            end
            if greedy_while_switch
                Wlen = greedy_augs[1](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
                if Wlen > Wlen_prev
                    if block ==0
                        J,I,_ = findnz(W)
                    else
                        I,J,_ = findnz(W)
                    end
                end
            end
            if verbose >= 1 && greedy_while_switch
                println("Wlen after Greedy:    ", Wlen)
            end
            
        elseif stage == 2 && (k > 1&& obj - last_obj > -params_list[stage].outerTOL && obj > last_obj * (1+params_list[stage].outerTOL) && Wlen_flag >= 5)
            break
        end
        
        if stage == 2
            if reduction_then_greedy && max(feval_flag, Wlen_flag) >= 5 ## for test
                if verbose >= 1
                    println("dropzero with Greedy Rule started...")
                    println("Wlen before dropping: ", Wlen)
                end
                lamb = dropzeros(lamb)
                if block ==0
                    J,I,_ = findnz(lamb)
                else
                    I,J,_ = findnz(lamb)
                end
                Wlen = length(lamb.nzval)
                Wlen_prev = Wlen
                W = sparse(J,I,fill(true,Wlen),n,n)
                Wlen = greedy_augs[end](phi,xi,Xflat,Y,W,Wlen,lamb,n,d)
                if Wlen > Wlen_prev
                    if block ==0
                        J,I,_ = findnz(W)
                    else
                        I,J,_ = findnz(W)
                    end
                end
                if verbose >= 1
                    println("Wlen after dropping:  ", Wlen_prev)
                    println("Wlen after Rule1:     ", Wlen)
                end
                feval_flag = 0;
                Wlen_flag = 0;
            end
        end
        Wlen_prev = Wlen;        
        last_obj = obj;
    end
    runtime = time() - start
    if verbose >= 1
        println("\nConvex Regression finishes in ",runtime," s.")
    end
    
    return phi,xi,lamb, W, Wlen, obj
end


