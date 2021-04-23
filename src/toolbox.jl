using LinearAlgebra, Random, SparseArrays,SIMD

ActiveSet = SparseMatrixCSC{Bool,Int64}
LambdaActiveSet = SparseMatrixCSC{T,Int64} where T<:AbstractFloat

index2d(k::Int64, d::Int64) =  (k-1)*d+1:k*d

function flat_to_mat(Xflat::Array{T,1},n::Int64,d::Int64) where T<:AbstractFloat
    return copy(reshape(Xflat,(d,n))')
end

function mat_to_flat(Xmat::Array{T,2},n::Int64,d::Int64) where T<:AbstractFloat
    return reshape(copy(Xmat'),n*d)
end

function idx2pair(idx,n)
    i = (idx-1)÷(n-1)+1;
    j = (idx-1)%(n-1)+1;
    j += (j>=i);
    return (i,j)
end

function A_dot_phi_plus_B_dot_xi!(res::Array{T,1},phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64,
        n::Int64,d::Int64) where {T<:AbstractFloat}
    @assert Wlen <= length(res)
    fill!(res,0)
    @simd for k in 1:Wlen
       @inbounds res[k] = (phi[J[k]] - phi[I[k]]) + dot(Xflat[index2d(I[k],d)]-Xflat[index2d(J[k],d)], xi[index2d(I[k],d)]);
    end
end

function A_dot_phi_plus_B_dot_xi(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64,
        n::Int64,d::Int64) where {T<:AbstractFloat}
    res = zeros(eltype(phi),Wlen);
    A_dot_phi_plus_B_dot_xi!(res, phi, xi, Xflat, I, J, Wlen, n, d)
    return res
end

function A_dot_phi_plus_B_dot_xi!(res::Array{T,1},phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},block_id::Int64,entry_ids::Array{Int64,1},len::Int64,n::Int64,d::Int64;block=0::Int) where {T<:AbstractFloat}
    @assert len <= length(res)
    fill!(res,0)
    if block == 0
        i = block_id
        @simd for k in 1:len
            @inbounds res[k] = (phi[entry_ids[k]] - phi[i]) + dot(Xflat[index2d(i,d)]-Xflat[index2d(entry_ids[k],d)], xi[index2d(i,d)]); 
        end   
    else
        j = block_id
        @simd for k in 1:len
            @inbounds res[k] = (phi[j] - phi[entry_ids[k]]) + dot(Xflat[index2d(entry_ids[k],d)]-Xflat[index2d(j,d)], xi[index2d(entry_ids[k],d)]); 
        end   
    end
end

function A_dot_phi_plus_B_dot_xi(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},block_id::Int64,entry_ids::Array{Int64,1},len::Int64,n::Int64,d::Int64;block=0::Int) where {T<:AbstractFloat}
    res = zeros(eltype(phi), len)
    A_dot_phi_plus_B_dot_xi!(res, phi, xi, Xflat, block_id, entry_ids, len, n, d, block=block)
    return res
end

function A_dot_phi!(res::Array{T,1},phi::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64) where {T<:AbstractFloat}
    @assert Wlen <= length(res)
    fill!(res,0)
    @simd for k in 1:Wlen
       @inbounds res[k] = (phi[J[k]] - phi[I[k]]);
    end
end
function A_dot_phi(phi::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64) where {T<:AbstractFloat}
    res = zeros(eltype(phi), Wlen)
    A_dot_phi!(res, phi, I, J, Wlen)
    return res
end

function B_dot_xi!(res::Array{T,1},xi::Array{T,1},Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, n::Int64,d::Int64) where {T<:AbstractFloat}
    @assert Wlen <= length(res)
    fill!(res,0)
    @simd for k in 1:Wlen
       @inbounds res[k] = dot(Xflat[index2d(I[k],d)]-Xflat[index2d(J[k],d)], xi[index2d(I[k],d)]);
    end
end
function B_dot_xi(xi::Array{T,1},Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, n::Int64,d::Int64) where {T<:AbstractFloat}
    res = zeros(eltype(phi), Wlen)
    B_dot_xi!(res, xi, Xflat, I, J, Wlen, n, d)
    return res
end


function AT_dot_lamb!(res::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},n::Int64) where T<:AbstractFloat
    @assert n <= length(res)
    fill!(res,0)
    @simd for k in 1:Wlen
        @inbounds res[I[k]]-=lamb[k];
        @inbounds res[J[k]]+=lamb[k];
    end
end

function AT_dot_lamb(I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},n::Int64) where T<:AbstractFloat
    res = zeros(eltype(lamb), n)
    AT_dot_lamb!(res,  I, J, Wlen, lamb, n)
    return res
end

function BT_dot_lamb!(res::Array{T,1},Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},n::Int64,d::Int64) where T<:AbstractFloat
    @assert n*d <= length(res)
    fill!(res,0)
    @simd for k in 1:Wlen
        @inbounds res[index2d(I[k],d)] += lamb[k] * (Xflat[index2d(I[k],d)]-Xflat[index2d(J[k],d)]);
    end
end

function BT_dot_lamb(Xflat::Array{T,1}, I::Array{Int64,1}, J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},n::Int64,d::Int64) where T<:AbstractFloat
    res = zeros(eltype(lamb),n*d)
    BT_dot_lamb!(res, Xflat, I, J, Wlen, lamb, n, d)
    return res
end

function Q_dot_lamb!(res::Array{T,1}, Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},
        n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}; if_subtract_Y=false::Bool) where T<:AbstractFloat
    AT_dot_lamb!(phi_tmp, I, J, Wlen, lamb,n)
    BT_dot_lamb!(xi_tmp, Xflat, I, J, Wlen, lamb,n,d)
    xi_tmp ./=rho
    if if_subtract_Y
        phi_tmp .-= Y
    end
    A_dot_phi_plus_B_dot_xi!(res, phi_tmp,xi_tmp,Xflat,I,J,Wlen,n,d)
end

function Q_dot_lamb(Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},n::Int64, d::Int64; if_subtract_Y=false::Bool) where T<:AbstractFloat
    res = zeros(eltype(Y),Wlen);
    phi_tmp = zeros(eltype(Y),n);
    xi_tmp = zeros(eltype(Y),n*d);
    Q_dot_lamb!(res, Xflat, Y, rho, I, J, Wlen, lamb, n, d, phi_tmp, xi_tmp, if_subtract_Y=if_subtract_Y);
    return res
end

function get_gradient_evaluation!(G::Array{T,1}, Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},
        n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}) where T<:AbstractFloat 
    Q_dot_lamb!(G, Xflat,Y,rho,I,J,Wlen,lamb,n,d,phi_tmp,xi_tmp,if_subtract_Y=true)
end


function power_iter(Xflat::Array{T,1},I::Array{Int64,1},J::Array{Int64,1},Wlen::Int64,rho::T,n::Int64,d::Int64;maxiter=300::Int64,TOL=1e-3::T) where T<:AbstractFloat
    rng = MersenneTwister(1234);
    x = randn(rng,Float64,Wlen);
    x /= norm(x);
    y = zeros(Wlen);
    nm = 0;
    phi_tmp = zeros(n);
    xi_tmp = zeros(n*d);
    for k in 1:maxiter
        Q_dot_lamb!(y, Xflat,Y,rho,I,J, Wlen, x, n, d, phi_tmp, xi_tmp, if_subtract_Y=false)
        if (abs(nm-norm(y))<TOL)
            break;
        end
        nm = norm(y);
        x .= y/nm;
    end

    return nm;
end

function get_objective_value(Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},
        n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}, s_tmp::Array{T,1}; get_phi_xi=false) where T<:AbstractFloat
    AT_dot_lamb!(phi_tmp, I, J, Wlen, lamb,n)
    BT_dot_lamb!(xi_tmp, Xflat, I, J, Wlen, lamb,n,d)
    res = 0.
    phi_tmp .-= Y
    phi_tmp .-= Y
    xi_tmp ./= rho
    
    A_dot_phi!(s_tmp, phi_tmp, I, J, Wlen)
    res += 0.5*dot(lamb,s_tmp)
    B_dot_xi!(s_tmp, xi_tmp, Xflat,I,J,Wlen,n,d)
    res += 0.5*dot(lamb,s_tmp)
    if get_phi_xi
        phi_tmp .+= Y
        phi_tmp .*= -1
        xi_tmp .*= -1
    end
    return res
end

function get_objective_value(Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1}, n::Int64, d::Int64) where T<:AbstractFloat
    obj = 0.5*dot(lamb, A_dot_phi(AT_dot_lamb(I,J,Wlen,lamb,n)-2Y,I,J,Wlen))+1/(2*rho)*dot(lamb,B_dot_xi(BT_dot_lamb(Xflat,I,J,Wlen,lamb,n,d),Xflat,I,J,Wlen,n,d));
    return obj
end

function get_objective_value(Xmat::Array{T,2},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1}, n::Int64, d::Int64) where T<:AbstractFloat
    return get_objective_value(mat_to_flat(Xmat,n,d),Y,rho,I,J, Wlen, lamb, n, d)
end

function get_objective_value_gradient!(G::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},
        n::Int64, d::Int64, phi_tmp::Array{T,1}, xi_tmp::Array{T,1}, s_tmp::Array{T,1}) where T<:AbstractFloat
    AT_dot_lamb!(phi_tmp, I, J, Wlen, lamb,n)
    BT_dot_lamb!(xi_tmp, Xflat, I, J, Wlen, lamb,n,d)
    res = 0;
    xi_tmp ./=rho
    phi_tmp .-= Y
    B_dot_xi!(s_tmp, xi_tmp, Xflat,I,J,Wlen,n,d)
    res += 0.5 * dot(lamb,s_tmp)
    G .= s_tmp
    A_dot_phi!(s_tmp, phi_tmp, I, J, Wlen)
    G .+= s_tmp
    phi_tmp .-=  Y
    A_dot_phi!(s_tmp, phi_tmp, I, J, Wlen)
    res += 0.5 * dot(lamb,s_tmp)
    
    return res
end


function check_feasibility(Xflat::Array{T,1},n::Int64,d::Int64,phi::Array{T,1},xi::Array{T,1}; TOL = 1.e-10::T) where T<:AbstractFloat
    min_constr = Inf
    constr = 0.
    for i in 1:n
        @simd for j in 1:n
            if j != i
                @inbounds constr = phi[j]-phi[i]+ dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
                if constr < -TOL
                    return false
                end
            end
        end
    end
    return true;
end


function compute_infeasibility(Xflat::Array{T,1},n::Int64,d::Int64,phi::Array{T,1},xi::Array{T,1}) where T<:AbstractFloat
    infeas = 0.
    max_viol = 0.
    constr = 0.
    for i in 1:n
        @simd for j in 1:n
            if j != i
                @inbounds constr = phi[j]-phi[i]+ dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
                infeas += min(constr,0)^2
                max_viol = min(max_viol,constr)
            end
        end
    end
    infeas = sqrt(infeas)/n
    return infeas,max_viol;
end



function get_primal_solution!(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1}, n::Int64, d::Int64) where T<:AbstractFloat
    AT_dot_lamb!(phi, I, J, Wlen, lamb,n)
    BT_dot_lamb!(xi, Xflat, I, J, Wlen, lamb,n,d)
    xi ./= (-rho)
    phi .-= Y
    phi .*= -1.
end



function get_primal_feasible_solution(Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},phi::Array{T,1},xi::Array{T,1}, n::Int64, d::Int64) where T<:AbstractFloat
    phi_f = similar(phi)
    xi_f = similar(xi)
    max_viol = 0.;
    viol::Float64 = 0.;
    nm_min::Float64 = Inf;
    nm::Float64 = 0.;
    argmin::Int64 = 0;
    mean_max_viol::Float64 = 0.;
    for j in 1:n
        max_viol = 0.;
        nm_min = norm(xi[index2d(j,d)]);
        argmin = j;
        for i in 1:n
            if i == j
                continue
            end
            @inbounds viol = phi[j] - phi[i] + dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
            if viol < max_viol
                max_viol = viol;
                argmin = i;
                nm_min = norm(xi[index2d(i,d)]);
            elseif viol == max_viol
                nm = norm(xi[index2d(i,d)]);
                if nm < nm_min
                    argmin = i;
                    nm_min = nm;
                end
            end
        end
        mean_max_viol += max_viol;
        phi_f[j] = phi[j] - max_viol;
        xi_f[index2d(j,d)] .= xi[index2d(argmin,d)];
    end
    mean_max_viol /= n;
    phi_f .+= mean_max_viol;
    return phi_f, xi_f
end



function get_duality_gap(Xflat::Array{T,1},Y::Array{T,1},rho::T,I::Array{Int64,1},J::Array{Int64,1}, Wlen::Int64, lamb::Array{T,1},phi::Array{T,1},xi::Array{T,1}, n::Int64, d::Int64; return_feasible_soln=false) where T<:AbstractFloat
    s_tmp = zeros(Wlen)
    L_upper = get_objective_value(Xflat,Y,rho,I,J, Wlen, lamb,n, d, phi, xi, s_tmp; get_phi_xi=true);
    phi_f,xi_f = get_primal_feasible_solution(Xflat, Y, rho, I, J, Wlen, lamb, phi, xi, n, d);
    L_lower = -(0.5*norm(phi_f-Y)^2+rho/2*norm(xi_f)^2)
    if return_feasible_soln
        return L_upper-L_lower, phi_f, xi_f
    else
        return L_upper-L_lower
    end
end


function partition_n_by_m(n::Int64,m::Int64)::Array{UnitRange{Int64},1}
    if n % m == 0
        partition = [(i-1)*m+1:i*m for i in 1:(n÷m)]
    else
        partition = [(i-1)*m+1:min(i*m,n) for i in 1:(n÷m+1)]
    end
    return partition
end


function compute_phi_minus_xi_dot_x(res::Array{T,1},Xflat::Array{T,1},phi::Array{T,1},xi::Array{T,1},n::Int64,d::Int64) where T<:AbstractFloat
    @assert length(res) == n
    fill!(res,0)
    @simd for i in 1:n
        @inbounds res[i] = phi[i] - dot(Xflat[index2d(i,d)],xi[index2d(i,d)])
    end
end



function cvx_predict!(y_pred::Array{T,1},Xtest::Array{T,2},phi::Array{T,1},Xi::Array{T,2}, n::Int64, d::Int64, piecevalues::Array{T,1},phi_minus_xi_dot_x::Array{T,1}) where T<:AbstractFloat
    ntest = size(Xtest,1)
    for j in 1:ntest
        max_ϕ = -Inf;
        piecevalues .= phi_minus_xi_dot_x
        piecevalues .+= Xi * Xtest[j,:]
        y_pred[j] = maximum(piecevalues);
    end
end


function cvx_predict(Xtest::Array{T,2},Xflat::Array{T,1},phi::Array{T,1},xi::Array{T,1}, n::Int64, d::Int64) where T<:AbstractFloat
    ntest =size(Xtest,1)
    y_pred = zeros(ntest)
    piecevalues = zeros(n)
    phi_minus_xi_dot_x = zeros(n)
    compute_phi_minus_xi_dot_x(phi_minus_xi_dot_x,Xflat,phi,xi,n,d)
    cvx_predict!(y_pred,Xtest,phi,copy(reshape(xi,(d,n))'),n,d,piecevalues,phi_minus_xi_dot_x)
    return y_pred
end




function rmse(y_pred, y)
    return norm(y_pred-y)/sqrt(length(y))
end

function rsquared(y_pred, y)
    y_mean = mean(y)
    return 1-sum((y_pred.-y).^2)/sum((y.-y_mean).^2)
end


