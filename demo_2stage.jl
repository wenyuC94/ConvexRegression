include("src/toolbox.jl")
include("src/augmentation.jl")
include("src/solver.jl")
include("src/cvxreg.jl")
include("src/testdata.jl")
using DataFrames, JLD,CSV,JSON
using DataStructures


n = 10000;
d = 4;
logrho = -3;
rho = 10.0^logrho;
func_type = "A";
random_state = 22;
ntrain = n;
ntest = 10000;
r = (d <= 3) ? 0.8 : 0.9;
SNR = 3;

LBFGS = false;
rule = 4;
LS = true;
WS = false;
inexact=false;

xtrain, ytrain, xtest, ytest, xtest_bd, ytest_bd, xtest_int, ytest_int = generate_training_test_samples(func_type,ntrain,ntest,d,r,SNR,1)

maxiter=1000
maxtime = 1800
verbose = 1
block = 0
Xmat = xtrain;
Y = ytrain;
Xflat = mat_to_flat(Xmat,n,d);
violTOL = 1.0e-4;
innerTOL = 1.0e-6;
outerTOL = 1.0e-4;
dropzero = true

params1 = AlgorithmParameters(inexact=true, violTOL=violTOL, outerTOL=outerTOL)
params2 = AlgorithmParameters(inexact=inexact, violTOL=1.0e-8, outerTOL=outerTOL)
aug1 = ActiveSetAugmentation(rule,n,block,params1.violTOL)
aug2 = ActiveSetAugmentation(rule,n,block,params2.violTOL)
runtime = @elapsed phi,xi,lamb, W, I,J, Wlen,obj = two_stage_active_set(Xmat,Y,rho; verbose = verbose,random_state=random_state, 
    maxiter=maxiter, block=block, dropzero=dropzero,  params_list = [params1, params2], augs = [aug1, aug2]);


if block == 0
    J,II,_= findnz(W)
else
    II,J,_= findnz(W)
end

pinfeas, max_viol = compute_infeasibility(Xflat,n,d,phi,xi)
gap,phi_f,xi_f = get_duality_gap(Xflat,Y,rho,II,J, Wlen, lamb.nzval,phi,xi, n, d,return_feasible_soln=true)

lamb_sparse = dropzeros(lamb)
if block ==0
    J_sparse,I_sparse,_ = findnz(lamb_sparse)
else
    I_sparse,J_sparse,_ = findnz(lamb_sparse)
end
Wlen_sparse = length(lamb_sparse.nzval)
W_sparse = sparse(J_sparse,I_sparse,fill(true,Wlen_sparse))

ypred_is = cvx_predict(xtrain, Xflat,phi_f,xi_f, n, d);
ypred = cvx_predict(xtest, Xflat,phi_f,xi_f, n, d);
ypred_bd = cvx_predict(xtest_bd, Xflat,phi_f,xi_f, n, d);
ypred_int = cvx_predict(xtest_int, Xflat,phi_f,xi_f, n, d);


rmse_is = rmse(ypred_is, ytrain)
rsq_is = rsquared(ypred_is, ytrain)
rmse_oos = rmse(ypred, ytest)
rsq_oos = rsquared(ypred, ytest)
rmse_bd = rmse(ypred_bd, ytest_bd)
rsq_bd = rsquared(ypred_bd, ytest_bd)
rmse_int = rmse(ypred_int, ytest_int)
rsq_int = rsquared(ypred_int, ytest_int)
