
import Pkg
Pkg.activate(".")
Pkg.instantiate()

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


# function gen_signal(len_time; y_0 = 1.0)

#     λ_true = rand(Beta(10., 4.))
#     τ_true = rand(Gamma(2., 1.))
#     σ_true = sqrt(inv(τ_true))

#     global signal = zeros(len_time)
#     signal[1] = y_0
#     for k in 2:len_time
#         global signal[k] = (1 -λ_true*Δt)*signal[k-1] + σ_true*rand(Normal(0,Δt))
#     end
#     return signal ./ std(signal)
# end

function gen_signal_Mat12(tsteps, λ, σ)

    N = length(tsteps)

    K = zeros(N,N)
    for (i,t) in enumerate(tsteps)
        for (j,ti) in enumerate(tsteps)
            K[i,j] = σ^2*exp( −λ*abs(t−ti) )
        end
    end

    f = rand()*2
    ϕ = rand()*π

    return rand(MvNormal(sin.(f.*tsteps .+ ϕ),K))
end


# Number of repetitions
num_reps = 10

# Time settings
Δt = 0.1
len_times = 2 .^collect(3:10)

# Optimization settings
max_iters = 1000

# Model parameters
M = 1
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.01
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

runtimes_MML = zeros(num_reps)
runtimes_HMC = zeros(num_reps)
runtimes_AR1 = zeros(num_reps)

@showprogress for (k,len_time) in enumerate(len_times)
    for rep in 1:num_reps

        λ_true = rand(Beta(10., 4.))
        τ_true = rand(Gamma(2., 1.))
        σ_true = sqrt(inv(τ_true))

        global tsteps = range(0, step=Δt, length=len_time)
        global signal = gen_signal_Mat12(tsteps, λ_true, σ_true)

        runtimes_MML[rep] = @elapsed optGP_Mat12(tsteps,signal, max_iters=max_iters)
        runtimes_AR1[rep] = @elapsed optAR1(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0)
        runtimes_HMC[rep] = @elapsed optMCMC_Mat12(tsteps,signal)
        
    end
    
    "Store results"
    lt = lpad(len_time, 5, '0')
    jldsave("experiments/simulation/results/runtimes-AR1-lentime$lt.jld2"; 
        tsteps, signal, len_time, runtimes_MML, runtimes_HMC, runtimes_AR1)
end