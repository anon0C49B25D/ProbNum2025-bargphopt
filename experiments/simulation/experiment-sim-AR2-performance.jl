
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


# Identities
experiment_ids = 1:10

# Time parameters
Δt = 0.1
len_trn_time = 100
len_tst_time = 100
trn_tsteps = range(0, step=Δt, length=len_trn_time)
tst_tsteps = range(0, step=Δt, length=len_tst_time)

# Optimization settings
max_iters = 1000

# Model parameters
M = 2
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.1
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

@showprogress for nn in experiment_ids

    "Generate signals"

    λ_true = rand(Beta(10., 4.))
    τ_true = rand(Gamma(2., 1.))
    σ_true = sqrt(inv(τ_true))

    global trn_signal = gen_signal_Mat32(trn_tsteps, λ_true, σ_true)
    global tst_signal = gen_signal_Mat32(tst_tsteps, λ_true, σ_true)

    """Optimize Gaussian processes"""

    params_MML = optGP_Mat32(trn_tsteps, trn_signal, max_iters=max_iters)
    runtime_MML = @belapsed optGP_Mat32(trn_tsteps, trn_signal, max_iters=max_iters)
    performance_MML = test_params_Mat32(params_MML[:ll], params_MML[:lσ], tst_tsteps, tst_signal)      

    """Infer hyperparameters"""

    params_AR2 = optAR2(trn_tsteps, trn_signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, Δ=Δt) 
    runtime_AR2 = @belapsed optAR2(trn_tsteps, trn_signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, Δ=Δt)
    performance_AR2 = test_params_Mat32(params_AR2[:ll], params_AR2[:lσ], tst_tsteps, tst_signal) 
    
    """Markov Chain Monte Carlo"""

    params_HMC = optMCMC_Mat32(trn_tsteps, trn_signal, max_iters=max_iters)    
    runtime_HMC = @belapsed optMCMC_Mat32(trn_tsteps, trn_signal, max_iters=max_iters)    
    performance_HMC = test_params_Mat32(params_HMC[:ll], params_HMC[:lσ], tst_tsteps, tst_signal) 

    "Save results"
    
    trialnum = lpad(nn, 3, '0')
    jldsave("experiments/simulation/results/experiment-AR2-trialnum$trialnum.jld2"; 
        trn_tsteps, tst_tsteps, trn_signal, tst_signal, λ_true, τ_true, Δt,
        performance_MML, params_MML, runtime_MML,
        performance_HMC, params_HMC, runtime_HMC, 
        performance_AR2, params_AR2, runtime_AR2)

end