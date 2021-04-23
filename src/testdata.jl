using Statistics,Random,LinearAlgebra,HDF5,JLD

function randb(rng::AbstractRNG,n::Int64, d::Int64, r::Float64=0.8, boundary::Union{Nothing,Bool}=nothing)
    if typeof(boundary) == Nothing
        return copy(rand(rng,d,n)')
    else
        @assert r>0 && r<1
        p = r^d
        if boundary
            N = ceil(Int, n/(1-p)*1.1)
            X = copy(rand(rng,d,N)')
            X = X[map(x->norm(x,Inf), eachslice(X, dims=1)).>r,:]
            X = X[1:n,:]
        else
            N = ceil(Int, n/p*1.1)
            X = copy(rand(rng,d,N)')
            X = X[map(x->norm(x,Inf), eachslice(X, dims=1)).<=r,:]
            X = X[1:n,:]
        end
        return X
    end
end


randb(n::Int64, d::Int64, r::Float64=0.8, boundary::Union{Nothing,Bool}=nothing) = randb(Random.default_rng(),n,d,r,boundary)

function generate_training_test_samples(func_type,ntrain,ntest,d,r,SNR,random_state =42;normalize=true)
    Random.seed!(random_state);
    xtrain = randb(ntrain, d, r, nothing)
    xtest = randb(ntest, d, r, nothing)
    xtest_bd = randb(ntest, d, r, true)
    xtest_int = randb(ntest, d, r, false)
    
    ytrain = zeros(ntrain);
    ytest = zeros(ntest);
    ytest_bd = zeros(ntest);
    ytest_int = zeros(ntest);
    
    if func_type == "A"
        ######## Type A ########
        ### phi(x) = norm(x,2)^2
        for (x,y) in zip([xtrain,xtest,xtest_bd,xtest_int],[ytrain,ytest,ytest_bd,ytest_int])
            y .= map(z->norm(z,2)^2, eachslice(x, dims=1))
        end
    elseif func_type == "B"
        ######## Type B ########
        ### phi(x) = max_j <a_j,x>
        a = 2 * rand(d,2*d) .- 1.
        for (x,y) in zip([xtrain,xtest,xtest_bd,xtest_int],[ytrain,ytest,ytest_bd,ytest_int])
            y .= map(z->maximum(a'*z), eachslice(x, dims=1))
        end
    elseif func_type == "C"
        ######## Type C ########
        ### phi(x) = norm(x,4)^4
        for (x,y) in zip([xtrain,xtest,xtest_bd,xtest_int],[ytrain,ytest,ytest_bd,ytest_int])
            y .= map(z->norm(z,4)^4, eachslice(x, dims=1))
        end
    elseif func_type == "D"
        ######## Type D ########
        ### phi(x) =  <a,x>
        a = 2*rand(d) .- 1
        for (x,y) in zip([xtrain,xtest,xtest_bd,xtest_int],[ytrain,ytest,ytest_bd,ytest_int])
            y .= map(z->dot(a,z), eachslice(x, dims=1))
        end
    end
    
    if normalize
        x_mean = mean(xtrain, dims = 1)
        x_norm = reshape(map(z->norm(z,2), eachslice(xtrain.-x_mean, dims=2)), (1,d))
        for x in [xtest, xtest_bd, xtest_int, xtrain]
            x .-= x_mean
            x ./= x_norm
        end
    end
    
    sigma = sqrt(var(ytrain)/SNR)
    for y in [ytest, ytest_bd, ytest_int, ytrain]
        y .+= sigma*randn(length(y))
    end
    
    if normalize
        y_mean = mean(ytrain)
        y_norm = norm(ytrain.-y_mean,2)
        for y in [ytest, ytest_bd, ytest_int, ytrain]
            y .-= y_mean
            y ./= y_norm
        end
    end
    
    return xtrain, ytrain, xtest, ytest, xtest_bd, ytest_bd, xtest_int, ytest_int
end


function load_dataset(fname)
    if endswith(fname, ".h5")
        file = h5open(fname, "r")
        xtrain = copy(read(file, "xtrain")["block0_values"]')
        xtest = copy(read(file, "xtest")["block0_values"]')
        xtest_bd = copy(read(file, "xtest_bd")["block0_values"]')
        xtest_int = copy(read(file, "xtest_int")["block0_values"]')
        ytrain = read(file, "ytrain")["values"]
        ytest = read(file, "ytest")["values"]
        ytest_bd = read(file, "ytest_bd")["values"]
        ytest_int = read(file, "ytest_int")["values"]
        close(file)
    elseif endswith(fname, ".jld")
        dict = load(fname)
        xtrain = dict["xtrain"]
        xtest = dict["xtest"]
        xtest_bd = dict["xtest_bd"]
        xtest_int = dict["xtest_int"]
        ytrain = dict["ytrain"]
        ytest = dict["ytest"]
        ytest_bd = dict["ytest_bd"]
        ytest_int = dict["ytest_int"]
    end
        
    return xtrain, ytrain, xtest, ytest, xtest_bd, ytest_bd, xtest_int, ytest_int
end
