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

xtrain, ytrain, xtest, ytest, xtest_bd, ytest_bd, xtest_int, ytest_int = generate_training_test_samples(func_type,ntrain,ntest,d,r,SNR,1)




maxiter= 10000000
maxtime = 43200
# maxiter = 300
# maxtime = 3600
verbose = 1
block = 0
rho = 10.0^logrho
Xmat = xtrain
Y = ytrain
Xflat = reshape(copy(Xmat'),n*d);


evaluation = :time
if n < 30000
    evaluation_freq = 600
elseif n < 50000
    evaluation_freq = 900
elseif n <= 100000
    evaluation_freq = 1200
end


violTOL =1.0e-4
outerTOL=0.
rule = 2
greedy_rule = 1
params1 = AlgorithmParameters(inexact=true, violTOL=violTOL, outerTOL=outerTOL)
params2 = AlgorithmParameters(inexact=false, violTOL=1.0e-8, outerTOL=outerTOL, decrease_L= true)
params_list = [params1, params2]
aug1 = ActiveSetAugmentation(rule,n,block,params1.violTOL)
aug2 = ActiveSetAugmentation(rule,n,block,params2.violTOL)
augs = [aug1, aug2]
greedy_augs = (greedy_rule != 0) ? [ActiveSetAugmentation(greedy_rule,n,block,params2.violTOL)] : Array{ActiveSetAugmentation,1}([])

reduction_while_switch = true
greedy_while_switch = Bool(greedy_rule != 0)
reduction_then_greedy = greedy_while_switch


measure = @timed phi,xi,lamb, W, Wlen, inner_profile_dict, outer_profile_dict, eval_KKT_dict, greedy_profile_dict = 
two_stage_active_set_profiling(Xmat, Y, rho; verbose=verbose, random_state=random_state,  maxiter=maxiter, maxtime=maxtime, block=block,
    reduction_while_switch=reduction_while_switch, greedy_while_switch=greedy_while_switch, 
    params_list=params_list, augs=augs, evaluation = evaluation, evaluation_freq =evaluation_freq, reduction_then_greedy=reduction_then_greedy,greedy_augs=greedy_augs)


if block == 0
    J,II,_= findnz(W)
else
    II,J,_= findnz(W)
end
KKT_eval_elapsed = @elapsed gap, pinfeas, max_viol, phi_f,xi_f,obj_ub, obj_lb = get_dual_solution_quality(Xflat,Y,rho,II,J, Wlen, lamb.nzval, phi, xi, n, d,return_feasible_soln=true)
push!(eval_KKT_dict["pinfeas"], pinfeas)
push!(eval_KKT_dict["max_viol"], max_viol)
push!(eval_KKT_dict["obj_ub"], obj_ub)
push!(eval_KKT_dict["obj_lb"], obj_lb)
push!(eval_KKT_dict["dual_gap"], gap)
push!(eval_KKT_dict["best_gap"], obj_ub - maximum(eval_KKT_dict["obj_lb"]))
push!(eval_KKT_dict["eval_time"], inner_profile_dict["time"][end])
push!(eval_KKT_dict["eval_outer"], length(outer_profile_dict["augmentation"]))
push!(eval_KKT_dict["eval_inner"], length(inner_profile_dict["objs"]))
push!(eval_KKT_dict["elapsed"], KKT_eval_elapsed)




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





