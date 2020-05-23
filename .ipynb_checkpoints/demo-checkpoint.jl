include("src/toolbox.jl")

include("src/cvxreg.jl")

include("src/testdata.jl")

#### specify problem
n = 2000; 
d = 2;
func_type = "A"; # for piecewise maximum function type in the paper, use func_type = "D" (in paper that's data_type = B)
logrho = -3;
SNR = 10;


#### specify algorithm
random_state = 22;
rule = 2;
heuristics = false; # whether apply heuristics for Rule 1
LS = true;
WS = false;
inexact = true;
#above is our best algorithm ASGD-Rule2-LS setting

## other inputs related to rules
P = 0;
M = 0;
K = 0;
G = 0;
J = 0;
block = 0; # blocking: 0 for \Omega_{i\dot}, 1 for \Omega_{\cdot j}

if inexact == true
    max_steps =5
else
    max_steps=3000
end

## some default setting for algorithm and rules
warm_start_partition="five"
if rule == -1
    heuristics = true
    J = 100
    P = 1
elseif rule == 1
    P = 1
elseif rule == 2
    K = n
elseif rule == 3
    P = 1
elseif rule == 4
    M = n*4
    K = n
elseif rule == 5
    G = Int(n/4)
    P = 4
end
verbose = 1;

#### specify stopping criterion
maxiter=3000;
maxtime = 10800;
violTOL =1.0e-3;
innerTOL=1.0e-5;
outerTOL=1.0e-4;


#### generate synthetic data
rho = 10.0^logrho;
Xmat, Y, _, _ = generate_examples(func_type,n,d,SNR,1);


#### training

## active set (recommended for n <= 5000)
all_objs,all_obj_times,initialize_time,searching_times,pow_times, optimization_starting_times,phi,xi,lamb,W,Wlen = active_set_timing(Xmat, Y, rho, max_steps,maxiter,violTOL , innerTOL, outerTOL, verbose,random_state,maxtime,LS, WS,inexact,rule,warm_start_partition,P,M,K,G,block,heuristics,J)
## active set (limited memory version) (recommended for n > 5000)
all_objs,all_obj_times,initialize_time,searching_times,pow_times, optimization_starting_times,phi,xi,lamb,W,Wlen = active_set_timing_limited_memory(Xmat, Y, rho, max_steps,maxiter,violTOL , innerTOL, outerTOL, verbose,random_state,maxtime,LS, WS,inexact,rule,warm_start_partition,P,M,K,G,block,heuristics,J)

#### duality gap evaluation (this part is not timed)
phi, xi, lamb, obj, gap, max_viol, num_tol_viol, num_viol, W, Wlen = get_duality_gap_partial(Xmat, Y, rho, W[1:Wlen], lamb, violTOL);