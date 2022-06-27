using StatsBase
include("toolbox.jl")



function augmentation_rule1(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,n::Int64,d::Int64,
        viol::Array{T,1},sort_index::Array{Int64,1},all_entries::Array{Int64,1}, entries::Array{Int64,1};
        TOL=1e-3::T,P=1::Int64,block=0::Int64) where {T <:  AbstractFloat}
    fill!(viol,0)
    fill!(entries,0)
    @simd for i in 2:n
        @inbounds all_entries[i-1]=i
    end
    P1::Int64 = 0;
    len::Int64 = 0;
    p::Int64=0;
    for i in 1:n
        len = 0
        if i != 1
            all_entries[i-1] -= 1
        end
        for j in all_entries
            if ~W[j,i]
                len += 1
                entries[len] = j
            end
        end
        A_dot_phi_plus_B_dot_xi!(viol,phi,xi,Xflat,i,entries,len,n,d,block=block);
        P1 = min(sum(viol.<-TOL),P);
        if P1 == 1
            _,p = findmin(viol);
            W[entries[p],i] = true
            lamb[entries[p],i] = -1e-16
            lamb[entries[p],i] = 0.0
            Wlen += 1
        elseif P1 > 1
            sortperm!(sort_index,viol;alg=PartialQuickSort(P1))
            for k in 1:P1
                W[entries[sort_index[k]],i] = true
                lamb[entries[sort_index[k]],i] = -1e-16
                lamb[entries[sort_index[k]],i] = 0.0
                Wlen += 1
            end
        end
    end
    return Wlen
end


function augmentation_rule2(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,
        n::Int64,d::Int64,samples::Array{Int64,1};TOL=1e-3,K=n,block=0) where T<:AbstractFloat
    i::Int64 = 0
    j::Int64 = 0
    viol::Float64 = 0.
    @assert length(samples)==K
    sample!(1:n*(n-1), samples,replace=false)
    for k in samples
        (i,j) = idx2pair(k, n)
        if block == 0 && W[j,i]
            continue
        elseif block == 1 && W[i,j]
            continue
        end
        @inbounds viol = phi[j] - phi[i] + dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
        if viol < -TOL
            if block == 0
                W[j,i] = true
                lamb[j,i] = -1e-16
                lamb[j,i] = 0.
                Wlen += 1
            else
                W[i,j] = true
                lamb[i,j] = -1e-16
                lamb[i,j] = 0.
                Wlen += 1
            end
        end
    end
    return Wlen
end

function augmentation_rule3(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,
        n::Int64,d::Int64,all_entries::Array{Int64,1}, samples::Array{Int64,1};TOL=1e-3,P=1,block=0) where T<:AbstractFloat
    i::Int64 = 0
    j::Int64 = 0
    viol::Float64 = 0.
    @simd for i in 2:n
        @inbounds all_entries[i-1] = i
    end
    @assert length(samples) == P
    for i in 1:n
        if i != 1
            all_entries[i-1] -= 1
        end
        sample!(all_entries, samples, replace=false)
        for j in samples
            if block == 0
                if W[j,i]
                    continue
                end
                @inbounds viol = phi[j] - phi[i] + dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
                if viol < -TOL
                    W[j,i] = true
                    lamb[j,i] = -1e-16
                    lamb[j,i] = 0.
                    Wlen += 1
                end
            else
                if W[j,i]
                    continue
                end
                @inbounds viol = phi[i] - phi[j] + dot(Xflat[index2d(j,d)]-Xflat[index2d(i,d)], xi[index2d(j,d)])
                if viol < -TOL
                    W[j,i] = true
                    lamb[j,i] = -1e-16
                    lamb[j,i] = 0.
                    Wlen += 1
                end
            end
        end
    end
    return Wlen
end

function augmentation_rule4(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,
        n::Int64,d::Int64,viol::Array{T,1},sort_index::Array{Int64,1},samples::Array{Int64,1}, I_tmp::Array{Int64,1}, J_tmp::Array{Int64,1};
        TOL=1e-3,M=4n, K=n,block=0) where T<:AbstractFloat
    i::Int64 = 0
    j::Int64 = 0
    fill!(viol,0);
    len = 0;
    total_viol = 0;
    @assert length(samples) == M
    
    
    sample!(1:n*(n-1), samples,replace=false)
    for k in samples
        (i,j) = idx2pair(k, n)
        if block == 0 && W[j,i]
            continue
        elseif block == 1 && W[i,j]
            continue
        end
        len += 1
        @inbounds viol[len] = phi[j] - phi[i] + dot(Xflat[index2d(i,d)]-Xflat[index2d(j,d)], xi[index2d(i,d)])
        I_tmp[len] = i;
        J_tmp[len] = j;
        if viol[len] < -TOL
            total_viol += 1;
        end
    end
    K1 = min(K,total_viol);
    if K1 != 0
        sortperm!(sort_index,viol;alg=PartialQuickSort(K1))
        for k in 1:K1
            i = I_tmp[sort_index[k]]
            j = J_tmp[sort_index[k]]
            Wlen += 1
            if block == 0
                W[j,i] = true
                lamb[j,i] = -1e-16
                lamb[j,i] = 0.0
            else
                W[i,j] = true
                lamb[i,j] = -1e-16
                lamb[i,j] = 0.0
            end
        end
    end
    return Wlen
end
    
function augmentation_rule5(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,n::Int64,d::Int64,
        viol::Array{T,1},sort_index::Array{Int64,1},blocks::Array{Int64,1}, all_entries::Array{Int64,1}, entries::Array{Int64,1};
        TOL=1e-3::T,G=n÷4::Int64,P=4::Int64,block=0::Int64) where {T <:  AbstractFloat}
    fill!(viol,0)
    fill!(entries,0)
    @assert length(blocks) == G
    sample!(1:n, blocks, replace=false, ordered=true)
    @simd for i in 2:n
        @inbounds all_entries[i-1]=i
    end
    P1::Int64 = 0;
    len::Int64 = 0;
    p::Int64=0;
    last_i::Int64 = 1;
    for i in blocks
        len = 0
        all_entries[last_i:i-1] .-= 1
        last_i = i;
        for j in all_entries
            if ~W[j,i]
                len += 1
                entries[len] = j
            end
        end
        A_dot_phi_plus_B_dot_xi!(viol,phi,xi,Xflat,i,entries,len,n,d,block=block);
        P1 = min(sum(viol.<-TOL),P);
        if P1 == 1
            _,p = findmin(viol);
            W[entries[p],i] = true
            lamb[entries[p],i] = -1e-16
            lamb[entries[p],i] = 0.0
            Wlen += 1
        elseif P1 > 1
            sortperm!(sort_index,viol;alg=PartialQuickSort(P1))
            for k in 1:P1
                W[entries[sort_index[k]],i] = true
                lamb[entries[sort_index[k]],i] = -1e-16
                lamb[entries[sort_index[k]],i] = 0.0
                Wlen += 1
            end
        end
    end
    return Wlen
end


struct ActiveSetAugmentation
    rule::Int;
    n::Int;
    P::Int;
    M::Int;
    K::Int;
    G::Int;
    block::Int;
    TOL::Float64;
    viol::Union{Nothing,Array{Float64,1}};
    sort_index::Union{Nothing, Array{Int64,1}};
    all_entries::Union{Nothing, Array{Int64,1}};
    entries::Union{Nothing, Array{Int64,1}};
    samples::Union{Nothing, Array{Int64,1}};
    I_tmp::Union{Nothing, Array{Int64,1}};
    J_tmp::Union{Nothing, Array{Int64,1}};
    ### dynamic
    dynamic::Bool;
    dynamic_variable::Symbol;
    increase_freq::Int64;
    increment::Int64;
    current_calls::Base.RefValue{Int64};
    current_variable::Base.RefValue{Int64};
    
    function ActiveSetAugmentation(rule, n, block, TOL;P=0, M=0,K=0,G=0,ε=-1,α=-1, multiplier=1,addall=false,dynamic=false,dynamic_variable=:P, increase_freq=0,increment=0)
        if ~dynamic
            dynamic_variable = :none
            increase_freq = 0;
            increment = 0;
        end
        current_calls = Ref{Int64}(0);
        current_variable = Ref{Int64}(0);
        if rule == 1
            if ~addall
                P = (P == 0) ? min(multiplier,n-1) : P;
            else
                P = n-1;
            end
            if dynamic
                dynamic_variable = :P
                increase_freq = (increase_freq == 0) ? 1 : increase_freq;
                increment = (increment == 0) ? 1 : increment;
            end
            viol = zeros(Float64, n-1)
            sort_index = zeros(Int64, n-1)
            all_entries = zeros(Int64, n-1)
            entries = zeros(Int64, n-1)
            samples = nothing;
            I_tmp = nothing;
            J_tmp = nothing;
        elseif rule == 2
            K = (K == 0) ? min(n*multiplier,n*(n-1)) : K;
            viol = nothing
            sort_index = nothing;
            all_entries = nothing;
            entries = nothing;
            samples = zeros(Int64,K);
            I_tmp = nothing;
            J_tmp = nothing;
        elseif rule == 3
            P = (P == 0) ? min(multiplier, n-1) : P;
            viol = nothing
            sort_index = nothing;
            all_entries = zeros(Int64, n-1);
            entries = nothing;
            samples = zeros(Int64, P);
            I_tmp = nothing;
            J_tmp = nothing;
        elseif rule == 4
            K = (K == 0) ? min(n*multiplier, n*(n-1)) : K;
            M = (M == 0) ? min(4*K, n*(n-1)) : max(M,K);
            viol = zeros(Float64,M);
            sort_index = zeros(Int64,M);
            all_entries = nothing;
            entries = nothing;
            samples = zeros(Int64,M);
            I_tmp = zeros(Int64, M);
            J_tmp = zeros(Int64, M);
        elseif rule == 5
            if ~addall
                P1 = (P == 0) ? 4 : P;
                G = (G == 0) ? n÷P1 : min(n÷P1, G);
                P = (P == 0) ? 4*multiplier : P1;
            else
                G = (G == 0) ? n÷4 : min(n÷4, G);
                P = n-1;
            end
            if dynamic
                dynamic_variable = :P
                increase_freq = (increase_freq == 0) ? 1 : increase_freq;
                increment = (increment == 0) ? 4 : increment;
            end
            viol = zeros(Float64, n-1)
            sort_index = zeros(Int64, n-1)
            all_entries = zeros(Int64, n-1)
            entries = zeros(Int64, n-1)
            samples = zeros(Int64, G);
            I_tmp = nothing;
            J_tmp = nothing;
        end
        new(rule, n, P, M, K, G, block,TOL,viol,sort_index,all_entries,entries, samples,I_tmp,J_tmp,
            dynamic,dynamic_variable,increase_freq,increment,current_calls,current_variable);
    end
end

function (aug::ActiveSetAugmentation)(phi::Array{T,1},xi::Array{T,1},Xflat::Array{T,1},Y::Array{T,1},W::ActiveSet,Wlen::Int64,lamb::LambdaActiveSet,n::Int64,d::Int64) where{T<:AbstractFloat}
    if aug.rule == 1
        P = aug.P;
        if aug.dynamic
            aug.current_calls[] += 1
            P = min(aug.current_variable[],n-1);
            if aug.current_calls[] % aug.increase_freq == 0
                aug.current_variable[] += aug.increment;
            end
        end
        return augmentation_rule1(phi,xi,Xflat,Y,W,Wlen,lamb,n,d,aug.viol,aug.sort_index,aug.all_entries, aug.entries;TOL=aug.TOL,P=P,block=aug.block)
        
    elseif aug.rule == 2
        return augmentation_rule2(phi,xi,Xflat,Y,W,Wlen,lamb,n,d,aug.samples;TOL=aug.TOL,K=aug.K,block=aug.block)
    elseif aug.rule == 3
        return augmentation_rule3(phi,xi,Xflat,Y,W,Wlen,lamb,n,d,aug.all_entries, aug.samples;TOL=aug.TOL,P=aug.P,block=aug.block)
    elseif aug.rule == 4
        return augmentation_rule4(phi,xi,Xflat,Y,W,Wlen,lamb,n,d,aug.viol,aug.sort_index,aug.samples,aug.I_tmp, aug.J_tmp;TOL=aug.TOL,M=aug.M,K=aug.K,block=aug.block)
    elseif aug.rule == 5
        P = aug.P;
        if aug.dynamic
            aug.current_calls[] += 1
            P = min(aug.current_variable[],n-1);
            if aug.current_calls[] % aug.increase_freq == 0
                aug.current_variable[] += aug.increment;
            end
        end
        return augmentation_rule5(phi,xi,Xflat,Y,W,Wlen,lamb,n,d,aug.viol,aug.sort_index,aug.samples,aug.all_entries,aug.entries;TOL=aug.TOL,G=aug.G,P=P,block=aug.block)
    end
end



function initial_active_set(Xmat,Y)
    n = size(Xmat,1);
    W = ifelse.(Y[1:n-1].>Y[2:n],[(i,i+1) for i in 1:n-1],[(i+1,i) for i in 1:n-1])
    return W
end


