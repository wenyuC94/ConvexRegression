include("toolbox.jl")
include("lbfgsb-wrapper.jl")

function cvxreg_qp_ineq_APG(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, n::Int64,d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; max_steps=1000, min_steps = 5, TOL=1e-4) where T<:AbstractFloat
    θ::Float64 = 1.0;
    θ_new::Float64 = 1.0;
    L = power_iter(Xflat,I,J,Wlen,rho,n,d);
    λ_x = copy(lamb.nzval);
    λ_y = copy(lamb.nzval);
    λ_new = zeros(eltype(Y),Wlen);
    s_tmp = zeros(eltype(Y),Wlen);
    grad = zeros(eltype(Y),Wlen);
    fnew = 0.;
    f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
    k = 0;
    for k in 1:max_steps
        get_gradient_evaluation!(grad, Xflat,Y,rho,I,J, Wlen, λ_y,n,d, phi_tmp, xi_tmp);
        λ_new .= min.(λ_y - grad/L,0);
        fnew = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
        if (fnew > f)
                θ = 1;
                λ_new .= λ_x;
                λ_y .= λ_x;
        end
        if (fnew <= f && fnew - f >-TOL && fnew>f*(1+TOL) && k >= min_steps)
                break
        end
        θ_new = (1+sqrt(1+4*θ^2))/2;
        γ = (θ-1)/θ_new;
        λ_y .= λ_new .+ γ*(λ_new .- λ_x);
        λ_x .= λ_new;
        θ = θ_new;
        f = fnew;
    end
    lamb.nzval .= λ_new
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    
    return fnew
end

function cvxreg_qp_ineq_APG_profiling(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, n::Int64,d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; max_steps=1000, min_steps = 5, TOL=1e-4) where T<:AbstractFloat
    all_times = zeros(Float64,max_steps+1);
    cur_time = @elapsed begin
        θ::Float64 = 1.0;
        θ_new::Float64 = 1.0;
        pow_time = @elapsed L = power_iter(Xflat,I,J,Wlen,rho,n,d);
        λ_x = copy(lamb.nzval);
        λ_y = copy(lamb.nzval);
        λ_new = zeros(eltype(Y),Wlen);
        s_tmp = zeros(eltype(Y),Wlen);
        grad = zeros(eltype(Y),Wlen);
        fnew = 0.;
        objs = zeros(Float64,max_steps+1);
        f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
        k = 0;
        iter = 0;
        objs[1] = f;
    end
    all_times[1] = cur_time;
    for k in 1:max_steps
        cur_time += @elapsed begin
            get_gradient_evaluation!(grad, Xflat,Y,rho,I,J, Wlen, λ_y,n,d, phi_tmp, xi_tmp);
            λ_new .= min.(λ_y - grad/L,0);
            fnew = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
        end
        objs[k+1] = fnew;
        all_times[k+1] = cur_time;
        iter += 1;
        cur_time += @elapsed begin
            if (fnew > f)
                    θ = 1;
                    λ_new .= λ_x;
                    λ_y .= λ_x;
            end
            if (fnew <= f && fnew - f >-TOL && fnew>f*(1+TOL)  && k >= min_steps)
                    break
            end
            θ_new = (1+sqrt(1+4*θ^2))/2;
            γ = (θ-1)/θ_new;
            λ_y .= λ_new .+ γ*(λ_new .- λ_x);
            λ_x .= λ_new;
            θ = θ_new;
            f = fnew;
        end
    end
    lamb.nzval .= λ_new
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    return fnew, all_times[1:iter+1], objs[1:iter+1], pow_time, L, iter
end




function cvxreg_qp_ineq_APG_ls(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, L::T, n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; 
        β_1 = 0.8, β_2 = 0.95, max_steps=1000::Int64, min_steps = 0::Int64, TOL=1e-4::T, maxls = 20::Int64, 
        L_init_estimate=:power, decrease_L = false, linesearch_type=:sufficient_decrease) where T<:AbstractFloat
    θ::Float64 = 1.0;
    θ_new::Float64 = 1.0;
    cur_ls = 0;
    λ_x = copy(lamb.nzval);
    λ_y = copy(lamb.nzval);
    λ_new = zeros(eltype(Y),Wlen);
    s_tmp = zeros(eltype(Y),Wlen);
    grad = zeros(eltype(Y),Wlen);
    f = 0.;
    fnew = 0.;
    grad_new = zeros(eltype(Y),Wlen);
    if L <= 0
        if L_init_estimate == :power
            L = power_iter(Xflat,I,J,Wlen,rho,n,d);
            f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
            cur_ls = maxls;
        else
            f = get_objective_value_gradient!(grad, Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
            λ_new = min.(λ_y - grad,0);
            fnew = get_objective_value_gradient!(grad_new, Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
            L = norm(grad_new.-grad,2)/norm(λ_new-λ_y,2);
        end
    else
        f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
    end
    ls_flag = false;
    k = 0;
    for k in 1:max_steps
        ls_flag = false;
        get_gradient_evaluation!(grad, Xflat,Y,rho,I,J, Wlen, λ_y,n,d, phi_tmp, xi_tmp);
        if decrease_L
            L *= ((k==1) ? β_1 : β_2);
        end
        while (true)
            λ_new .= min.(λ_y - grad/L,0);
            if linesearch_type == :sufficient_decrease
                fnew = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
            else
                fnew = get_objective_value_gradient!(grad_new, Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
            end
            if cur_ls >= maxls
                ls_flag = true
            elseif linesearch_type == :sufficient_decrease 
                ls_flag = (fnew <= f + dot(grad, λ_new - λ_y) + L/2 * norm(λ_new - λ_y,2)^2)
            elseif linesearch_type == :inner_product_condition
                if (fnew - f) > 0.01*max(abs(fnew), abs(f), 1) 
                    ls_flag = (fnew <= f + dot(grad, λ_new - λ_y) + L/2 * norm(λ_new - λ_y,2)^2)
                else
                    ls_flag = (abs(dot(λ_new - λ_y, grad_new - grad)) <= L/2 * norm(λ_new - λ_y,2)^2)
                end
            end
            if ls_flag
                cur_ls = 0
                break
            else
                L /= ((k==1) ? β_1 : β_2);
                cur_ls += 1
                if cur_ls >= maxls
                    L = power_iter(Xflat,I,J,Wlen,rho,n,d,maxiter=50);
                end
            end
        end
        if (fnew > f)
            θ = 1;
            λ_new .= λ_x;
            λ_y .= λ_x;
        end
        if (fnew <= f && fnew - f >-TOL && fnew>f*(1+TOL)  && k >= min_steps)
            break
        end
        θ_new = (1+sqrt(1+4*θ^2))/2;
        γ = (θ-1)/θ_new;
        λ_y .= λ_new .+ γ*(λ_new .- λ_x);
        λ_x .= λ_new;
        θ = θ_new;
        f = fnew;
    end
    lamb.nzval .= λ_new
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    
    return L, fnew
end



function cvxreg_qp_ineq_APG_ls_profiling(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, L::T, n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; 
        β_1 = 0.8, β_2 = 0.95, max_steps=1000::Int64, min_steps=0::Int64, TOL=1e-4::T, maxls = 20::Int64, 
        L_init_estimate=:power, decrease_L = false, linesearch_type=:sufficient_decrease) where T<:AbstractFloat
    all_times = zeros(Float64,max_steps+1);
    cur_time = @elapsed begin 
        θ::Float64 = 1.0;
        θ_new::Float64 = 1.0;
        cur_ls = 0;
        λ_x = copy(lamb.nzval);
        λ_y = copy(lamb.nzval);
        λ_new = zeros(eltype(Y),Wlen);
        s_tmp = zeros(eltype(Y),Wlen);
        grad = zeros(eltype(Y),Wlen);
        fnew = 0.;
        grad_new = zeros(eltype(Y),Wlen);
        f = 0.;
        if L <= 0
            if L_init_estimate == :power
                L = power_iter(Xflat,I,J,Wlen,rho,n,d);
                f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
                cur_ls = maxls;
            else
                f = get_objective_value_gradient!(grad, Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
                λ_new = min.(λ_y - grad,0);
                fnew = get_objective_value_gradient!(grad_new, Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
                L = norm(grad_new.-grad,2)/norm(λ_new-λ_y,2);
            end
        else
            f = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_x,n, d, phi_tmp, xi_tmp, s_tmp);
        end
        ls_flag = false;
        objs = zeros(Float64,max_steps+1);
        f_evals = zeros(Float64,max_steps);
        Ls = zeros(Float64,max_steps);
        k = 0;
        iter = 0;
        objs[1] = f;
    end
    all_times[1] = cur_time;
    for k in 1:max_steps
        cur_time += @elapsed begin
            get_gradient_evaluation!(grad, Xflat,Y,rho,I,J, Wlen, λ_y,n,d, phi_tmp, xi_tmp);
            if decrease_L
                L *= ((k==1) ? β_1 : β_2);
            end
            while (true)
                λ_new .= min.(λ_y - grad/L,0);
                if linesearch_type == :sufficient_decrease
                    fnew = get_objective_value(Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
                else
                    fnew = get_objective_value_gradient!(grad_new, Xflat,Y,rho,I,J, Wlen, λ_new,n, d, phi_tmp, xi_tmp, s_tmp);
                end
#                 println(L," ", f + dot(grad, λ_new - λ_y) + L/2 * norm(λ_new - λ_y,2)^2-fnew, " ", L/2 * norm(λ_new - λ_y,2)^2 -abs(dot(λ_new - λ_y, grad_new - grad)))
                if cur_ls >= maxls
                    ls_flag = true
                elseif linesearch_type == :sufficient_decrease 
                    ls_flag = (fnew <= f + dot(grad, λ_new - λ_y) + L/2 * norm(λ_new - λ_y,2)^2)
                elseif linesearch_type == :inner_product_condition
                    if abs(fnew - f) > 0.001*max(abs(fnew), abs(f), 1) 
                        ls_flag = (fnew <= f + dot(grad, λ_new - λ_y) + L/2 * norm(λ_new - λ_y,2)^2)
                    else
                        ls_flag = (abs(dot(λ_new - λ_y, grad_new - grad)) <= L/2 * norm(λ_new - λ_y,2)^2)
                    end
                end
                if ls_flag
                    f_evals[k] = cur_ls+1;
                    cur_ls = 0
                    break
                else
                    L /= ((k==1) ? β_1 : β_2);
                    cur_ls += 1
                    if cur_ls >= maxls
                        L = power_iter(Xflat,I,J,Wlen,rho,n,d,maxiter=50);
                    end
                end
            end
        end
        Ls[k] = L;
        objs[k+1] = fnew;
        all_times[k+1] = cur_time;
        iter += 1;
        cur_time += @elapsed begin
            if (fnew > f)
                θ = 1;
                λ_new .= λ_x;
                λ_y .= λ_x;
            end
            if (fnew <= f && fnew - f >-TOL && fnew>f*(1+TOL)  && k >= min_steps)
                break
            end
            θ_new = (1+sqrt(1+4*θ^2))/2;
            γ = (θ-1)/θ_new;
            λ_y .= λ_new .+ γ*(λ_new .- λ_x);
            λ_x .= λ_new;
            θ = θ_new;
            f = fnew;
        end
    end
    lamb.nzval .= λ_new
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    
    return L, fnew, all_times[1:iter+1], objs[1:iter+1], f_evals[1:iter], Ls[1:iter], iter
end

function cvxreg_qp_ineq_LBFGS(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; 
        max_steps=1000::Int64, TOL=1e-4::T) where T<:AbstractFloat
    s_tmp = zeros(eltype(Y),Wlen);
    f(G,λ) = get_objective_value_gradient!(G,Xflat,Y,rho,I,J, Wlen, λ,n, d, phi_tmp, xi_tmp, s_tmp)
    obj, minimizer = lbfgsb(f,lamb.nzval; ub=zeros(Wlen), ftol=TOL, maxiter=max_steps, profiling=false);
    lamb.nzval .= minimizer
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    return obj
end

function cvxreg_qp_ineq_LBFGS_profiling(Xflat::Array{T,1}, Y::Array{T,1}, rho::T, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, 
        lamb::LambdaActiveSet, n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; 
        max_steps=1000::Int64, TOL=1e-4::T) where T<:AbstractFloat
    all_times = zeros(Float64, 1+max_steps)
    cur_time = @elapsed begin
        s_tmp = zeros(eltype(Y),Wlen);
        f(G,λ) = get_objective_value_gradient!(G,Xflat,Y,rho,I,J, Wlen, λ,n, d, phi_tmp, xi_tmp, s_tmp)
    end
    all_times[1] = cur_time
    total_time = @elapsed obj,minimizer,objs,f_evals,iter = lbfgsb(f,lamb.nzval; ub=zeros(Wlen), ftol=TOL, maxiter=max_steps, profiling = true);
    all_times[2:iter+1] = cumsum(f_evals)/sum(f_evals)*total_time
    lamb.nzval .= minimizer
    phi_tmp .+= Y
    phi_tmp .*= -1
    xi_tmp .*= -1
    return obj, all_times[1:iter+1], objs, f_evals, iter
end