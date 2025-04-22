
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using BenchmarkTools
using Distributions
using LinearAlgebra
using DataFrames
using CSV
using JLD2
using ProgressMeter
using Optim
using GaussianProcesses
includet("../../optimization.jl");
includet("../../evaluation.jl");


# Load data
data_VS1 = DataFrame(CSV.File("experiments/monitorhydraulic/data/VS1.txt", header=false))
data_TS1 = DataFrame(CSV.File("experiments/monitorhydraulic/data/TS1.txt", header=false))
data_CP1 = DataFrame(CSV.File("experiments/monitorhydraulic/data/CP.txt", header=false))
num_cycles, num_points = size(data_VS1)
data = (data_VS1, data_TS1, data_CP1)

# Identities
experiment_ids = 1:100

# Time parameters
Δt = 1.0
len_trn_time = num_points
len_tst_time = num_points
trn_tsteps = range(0, step=Δt, length=len_trn_time)
tst_tsteps = range(0, step=Δt, length=len_tst_time)

# Optimization settings
max_iters = 1000

# Model parameters
M = 1
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.1
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

@showprogress for nn in experiment_ids

    "Load data set"

    # Randomly sample sensor
    rsensor = sample(1:3)

    # Randomly sample cycles
    rcycle_trn, rcycle_tst = sample(1:num_cycles, 2, replace=false)
    
    global trn_signal = collect(data[rsensor][rcycle_trn,:])
    global tst_signal = collect(data[rsensor][rcycle_tst,:])

    """Maximizing marginal likelihood"""

    try
        global params_MML = optGP_Mat12(trn_tsteps,trn_signal, max_iters=max_iters)
        global runtime_MML = @belapsed optGP_Mat12(trn_tsteps,trn_signal, max_iters=max_iters)
        global performance_MML = test_params_Mat12(params_MML[:ll], params_MML[:lσ], tst_tsteps, tst_signal)    
    catch
        global params_MML = (missing,missing)
        global runtime_MML = missing
        global performance_MML = (missing,missing,missing)
    end

    """Bayesian autoregression"""

    params_AR1 = optAR1(trn_tsteps,trn_signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0)    
    runtime_AR1 = @belapsed optAR1(trn_tsteps,trn_signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0)
    performance_AR1 = test_params_Mat12(params_AR1[:ll], params_AR1[:lσ], tst_tsteps, tst_signal) 
    
    """Markov Chain Monte Carlo"""

    params_HMC = optMCMC_Mat12(trn_tsteps, trn_signal)    
    runtime_HMC = @belapsed optMCMC_Mat12(trn_tsteps, trn_signal)
    performance_HMC = test_params_Mat12(params_HMC[:ll], params_HMC[:lσ], tst_tsteps, tst_signal) 
    
    "Save results"

    trialnum = lpad(nn, 3, '0')
    jldsave("experiments/monitorhydraulic/results/experiment-mhydr-AR1-trialnum$trialnum.jld2"; 
        trn_tsteps, tst_tsteps, trn_signal, tst_signal,
        performance_MML, params_MML, runtime_MML,
        performance_HMC, params_HMC, runtime_HMC, 
        performance_AR1, params_AR1, runtime_AR1)

end