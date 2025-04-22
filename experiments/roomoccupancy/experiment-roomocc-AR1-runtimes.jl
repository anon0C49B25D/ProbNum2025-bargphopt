
# import Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using Revise
using BenchmarkTools
using Distributions
using LinearAlgebra
using JLD2
using ProgressMeter
using Optim
using GaussianProcesses
includet("../../optimization.jl");
includet("../../evaluation.jl");



# Number of repetitions
num_reps = 10

# Time settings
Δt = 0.1
len_times = 2 .^collect(3:10)

# Optimization settings
max_iters = 1000

# Initial condition
u_0 = 1.0

# Model parameters
M = 1
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.01
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

runtimes_GP  = zeros(num_reps)
runtimes_AR1 = zeros(num_reps)

@showprogress for (k,len_time) in enumerate(len_times)
    for rep in 1:num_reps

        "Generate data"
        tsteps = range(0, step=Δt, length=len_time)
        signal = gen_signal(len_time; y_0 = 1.0)

        "GP"
        runtimes_GP[rep]  = @elapsed optGP_Mat12(tsteps,signal, max_iters=max_iters)

        "AR"
        runtimes_AR1[rep] = @elapsed optAR1(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0)
        
    end
    
    "Store results"
    lt = lpad(len_time, 5, '0')
    jldsave("experiments/roomoccupancy/results/runtimes-room-occ-AR1-lentime$lt.jld2"; 
        tsteps, signal, len_time, runtimes_GP, runtimes_AR1)
end