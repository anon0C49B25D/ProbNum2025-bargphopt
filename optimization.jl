using Optim
using ForwardDiff
using LinearAlgebra
using GaussianProcesses

includet("ARModels.jl"); using .ARModels
includet("RTSSmoothers.jl"); using .RTSSmoothers


function optGP_Mat12(tsteps, signal; max_iters=1)

    ll0 = 0.0
    lσ0 = 0.0
    
    kernel = Mat12Iso(ll0, lσ0)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)

    optimize!(gp, noise=false, domean=false, kern=true, lik=false, iterations=max_iters)

    return Dict(:ll => log(gp.kernel.ℓ), :lσ => log(sqrt(gp.kernel.σ2)))    
end

function optGP_Mat32(tsteps, signal; max_iters=1)

    ll0 = 0.0
    lσ0 = 0.0
    
    kernel = Mat32Iso(ll0, lσ0)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)

    optimize!(gp, noise=false, domean=false, kern=true, lik=false, iterations=max_iters)

    return Dict(:ll => log(gp.kernel.ℓ), :lσ => log(sqrt(gp.kernel.σ2)))
end

function optMCMC_Mat12(tsteps, signal; max_iters=1)

    kernel = Mat12Iso(0., 0.)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)
    
    set_priors!(kernel, [Exponential(), Exponential()])
    chain = mcmc(gp, noise=false, domean=false, kern=true, lik=false, nIter=max_iters)
    mnch = mean(chain, dims=2)

    return Dict(:ll => mnch[2], :lσ => mnch[3])
end

function optMCMC_Mat32(tsteps, signal; max_iters=1)

    kernel = Mat32Iso(0.5, 0.5)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)
    
    set_priors!(kernel, [Exponential(), Exponential()])
    chain = mcmc(gp, noise=false, domean=false, kern=true, lik=false, nIter=max_iters)
    mnch = mean(chain, dims=2)

    return Dict(:ll => mnch[2], :lσ => mnch[3])
end

function optTGP_Mat12(tsteps, signal, state0; max_iters=1)

    function J(hparams)
        expλ = exp(hparams[1])
        expσ = exp(hparams[2])
        A = [-expλ]
        Q = [2*expλ*expσ^2]
        C = [1.0]
        R = [1e-8]
        model = RTSSmoother(A,C,Q,R,state0)
        return log_marginal_likelihood(model,signal)
    end

    opt = Optim.Options(g_tol = 1e-12,
                        iterations = max_iters,
                        store_trace = false,
                        show_trace = false,
                        show_warnings = true)
    res = optimize(J, zeros(2), LBFGS(), opt)
    mins = Optim.minimizer(res)

    return Dict(:ll => mins[1], :lσ => mins[2])
end

function optTGP_Mat32(tsteps, signal, state0; max_iters=1)

    function J(hparams)
        expλ = exp(hparams[1])
        expσ = exp(hparams[2])
        A = [0.0        0.0;
            -expλ^2  -2*expλ]
        Q = [2*expλ*expσ^2]
        C = [1.0]
        R = [1e-8]
        model = RTSSmoother(A,C,Q,R,state0)
        return log_marginal_likelihood(model,signal)
    end

    opt = Optim.Options(g_tol = 1e-12,
                        iterations = max_iters,
                        store_trace = false,
                        show_trace = false,
                        show_warnings = true)
    res = optimize(J, zeros(2), LBFGS(), opt)
    mins = Optim.minimizer(res)

    return Dict(:ll => mins[1], :lσ => mins[2])
end

function optAR1(tsteps, signal; μ0=1.0, Λ0=[1.0], α0=2.0, β0=1/2, Δ=1.0)

    model = ARModel(μ0,Λ0,α0,β0, order=1)
    for y_k in signal
        ARModels.update!(model, y_k)
    end

    # Bound estimate
    if model.μ[1] >= 1; model.μ[1] = .99999; end

    # Reverting variable substitution
    ll_hat = log( Δ./(1 - model.μ[1]))
    lσ_hat = log( sqrt( model.β./( 2*Δ^2*(model.α - 1)*(1 - model.μ[1]) ) ) )

    return Dict(:ll => ll_hat, :lσ => lσ_hat)
end

function optAR2(tsteps, signal; μ0=-0.1, Λ0=[1.0], α0=2.0, β0=1/2, Δ=1.0, verbose=false, max_iters=1)

    model = ARModel(μ0,Λ0,α0,β0, order=2)
    for y_k in signal
        ARModels.update!(model, y_k)
    end

    opts = Optim.Options(time_limit=10., 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=max_iters)

    J(x) = (2 + 2*x[1]*Δ - model.μ[1])^2 + (-1-2*x[1]*Δ-x[1]^2*Δ^2 - model.μ[2])^2 + (1/(4*x[2]^2*x[1]^3*Δ^5) - (model.α-1)/model.β)^2
    results = Optim.optimize(J, 1e-8, Inf, [0.1,0.1], Fminbox(LBFGS()), opts, autodiff=:forward)
    λ_hat, σ_hat = Optim.minimizer(results)

    # Reverting variable substitution
    ll_hat = log( 1 ./λ_hat )
    lσ_hat = log( σ_hat )

    return Dict(:ll => ll_hat, :lσ => lσ_hat)
end