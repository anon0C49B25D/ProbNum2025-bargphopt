
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


function gen_signal_Mat32(tsteps, λ, σ)

    N = length(tsteps)

    K = zeros(N,N)
    for (i,t) in enumerate(tsteps)
        for (j,ti) in enumerate(tsteps)
            K[i,j] = σ^2*(1 + sqrt(3)*λ*abs(t - ti))*exp( −sqrt(3)*λ*abs(t - ti) )
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
M = 2
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.01
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

runtimes_MML = Vector(undef,num_reps)
runtimes_HMC = Vector(undef,num_reps)
runtimes_AR2 = Vector(undef,num_reps)

@showprogress for (k,len_time) in enumerate(len_times)
    for rep in 1:num_reps

        λ_true = rand(Beta(10., 4.))
        τ_true = rand(Gamma(2., 1.))
        σ_true = sqrt(inv(τ_true))

        global tsteps = range(0, step=Δt, length=len_time)
        global signal = gen_signal_Mat32(tsteps, λ_true, σ_true)

        try
            runtimes_MML[rep] = @elapsed optGP_Mat32(tsteps,signal, max_iters=max_iters)
        catch
            runtimes_MML[rep] = missing
        end
        try
            runtimes_AR2[rep] = @elapsed optAR2(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, Δ=Δt, max_iters=max_iters)
        catch
            runtimes_AR2[rep] = missing
        end
        try
            runtimes_HMC[rep] = @elapsed optMCMC_Mat32(tsteps,signal, max_iters=max_iters)
        catch
            runtimes_HMC[rep] = missing
        end
        
    end
    
    "Store results"
    lt = lpad(len_time, 5, '0')
    jldsave("experiments/simulation/results/runtimes-AR2-lentime$lt.jld2"; 
        tsteps, signal, len_time, runtimes_MML, runtimes_HMC, runtimes_AR2)
end