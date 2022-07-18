# Includes code for optimization and annealing of DENs

include("networks.jl")

using SparseArrays
using LinearAlgebra
using Zygote
using Statistics

""" Compute the constrained degrees of freedom and the
load for a uniform gravitational force F = -g ̂z.
"""
function solve_gravity(den::DiscreteElasticNetwork, k)
    n = den.netw.N_v
    H = network_hessian(den, k)

    # gravitational force
    g = -vcat(zeros(2n), ones(n))/n

    # solve for projected displacements
    g = den.Φ'*g

    y = H \ g

    y, g
end

""" Thermalize the elastic constants k
by convolving with Gaussian kernel of width σ using the
matrix dist of squared distances,
then applying multiplicative noise wit strength scaled
by 0.5/γ, and finally normalizing using the cost
constraint ∑_e k_e^γ = K.
"""
function thermalize(k, σ, γ, K, dist)
    # Smoothen
    k_therm = exp.(-dist./(2σ^2))*k

    # Multiplicative noise
    k_therm .*= exp.(0.5(0.5/γ)*randn(length(k)))

    # normalize
    k_therm./(sum(k_therm.^γ)/K).^(1/γ)
end

function estimate_σ0(netw::MechanicalNetwork)
    maximum(netw.node_pos[:,1]) - minimum(netw.node_pos[:,1])
end

""" Use a fixed-point iteration scheme to minimize the constrained
mechanical compliance.
If desired, uses a Simulated Annealing derived scheme to approximate
a global minimum.
"""
function optimize_iterate(den::DiscreteElasticNetwork, k0::Array, k_base,
        γ::Float64; iters=1600, θ=1e-6, anneal=false, thermalize_every=15,
        verbose=false, show_every=1, callback=nothing, mode=:angles,
        solve_fun=solve_gravity)

    function ssr(k)
        # mechanical compliance
        u, g = solve_fun(den, k_base .+ k)
        u'*g
    end

    function J_zygote(k)
        # use zygote to get gradient
        Zygote.gradient(ssr, k)[1]
    end

    function J_angles(k)
        # use the angle formula to get gradient
        y, g = solve_fun(den, k .+ k_base)
        network_edge_angles(den, k .+ k_base, y)
    end

    netw = den.netw
    dists = squared_bond_dists(netw)
    k_cur = copy(k0)

    σ = estimate_σ0(netw)
    f = (0.1/σ)^(thermalize_every/(0.5iters))

    if mode == :zygote
        J_fun = J_zygote
    else
        J_fun = J_angles
    end

    its = 0
    Jsigns = 1
    for i=1:iters
        its += 1
        # thermalize current iterate if so desired, use final 50% of iterations to converge
        thermalize_now = anneal && (mod(i, thermalize_every) == 0) && i < (0.5iters)
        if thermalize_now
            if verbose
                println("Thermalizing with σ=$σ")
            end

            k_cur = thermalize(k_cur, σ, γ, netw.N_e, dists)
            σ = f*σ
        end

        # relaxed fixed point iterations
        J = J_fun(k_cur)
        k_new = ((k_cur.^2).*abs.(J)).^(1/(γ+1))
        k_new = k_new./(sum(k_new.^γ)/netw.N_e).^(1/γ)

        if callback != nothing
            callback(i, k_new, thermalize_now)
        end

        Δnorm = norm(k_new - k_cur)
        oldnorm = norm(k_cur)
        Jsigns = sign(minimum(J)*maximum(J))

        if verbose && mod(i-1, show_every) == 0
            @show i, sign(minimum(J)*maximum(J)), Δnorm/oldnorm
        end

        normcheck = Δnorm < θ*oldnorm
        if (!anneal & normcheck) | (anneal & (i > 0.5iters) & normcheck)
            break
        end

        k_cur = copy(k_new)
    end

    # check that we have satisfied the convergence criteria
    # and that the gradient signs are all identical, such that our iteration
    # scheme was valid
    converged = (its < iters) && Jsigns == 1
    # return minimizer and minimum compliance
    k_cur, ssr(k_cur), its, converged
end


### Code for self-loads
function g_mass(den::DiscreteElasticNetwork, k; α=.5, β=0.2)
    # loads proportional to avg
    n = den.netw.N_v
    g = zeros(2n)

    # compute masses
    [g; [(-(1 - β)/n - β*mean(k[den.netw.neighbor_edges[i]].^α)/n) for i=1:n]]
end

function g_mass_frac(den::DiscreteElasticNetwork, k; α=.5, β=0.2)
    # compute mass fraction vein mass/total mass
    n = den.netw.N_v

    g = g_mass(den, k; α=α, β=β)[2n+1:end]
    m_vein = -sum(g .+ (1 - β)/n)
    m_total = -sum(g)

    m_vein/m_total
end

""" Compute the constrained degrees of freedom and the
load for a uniform gravitational force F = -g ̂z.
"""
function solve_mass(den::DiscreteElasticNetwork, k; α=0.5, β=0.2)
    n = den.netw.N_v
    H = network_hessian(den, k)

    g = g_mass(den, k; α=α, β=β)
    # solve for projected displacements
    g = den.Φ'*g

    y = H \ g

    y, g
end

function objective_mass(den::DiscreteElasticNetwork, k; α=0.5, β=0.2)
    y, g = solve_mass(den, k; α=α, β=β)

    y'*g
end

function mass_objective_grads(den::DiscreteElasticNetwork, k, k_base; α=0.5, β=0.2)
    ## Compute the gradient of the objective with self-loads

    # gradient of the Hessian part
    y, g = solve_mass(den, k .+ k_base; α=α, β=β)
    G_H = network_edge_angles(den, k .+ k_base, y)

    # gradient of the load part
    k_g = α*(k .+ k_base).^(α-1)

    n = den.netw.N_v
    # Jacobian of the load vector (maybe we can speed this up? but know only g, need Φ'g)
    Jg = zeros(3n, den.netw.N_e)
    for i=1:n
        d = length(den.netw.neighbor_edges[i])
        for e in den.netw.neighbor_edges[i]
            Jg[2n+i,e] = -β*k_g[e]/d/n
        end
    end

    G_H, 2Jg'*den.Φ*y
end

function optimize_iterate_mass(den::DiscreteElasticNetwork, k0::Array, k_base,
        γ::Float64, α::Float64, β::Float64; iters=1600, θ=1e-6, anneal=false, thermalize_every=15,
        verbose=false, show_every=1, callback=nothing)

    function iterate_step_mass(den, k, k_base, γ, α, β)
        # take a step using the new iteration for self-loads
        G_H, G_f = mass_objective_grads(den, k, k_base; α=α, β=β)

        λγ = mean(k.*(G_H - G_f))

        denom = λγ .+ (k.^(1.0 - γ)).*G_f
        numer = (k.^2).*G_H

        numer./denom
    end

    netw = den.netw
    dists = squared_bond_dists(netw)
    k_cur = copy(k0)

    σ = estimate_σ0(netw)
    f = (0.1/σ)^(thermalize_every/(0.5iters))

    its = 0
    Jsigns = 1
    for i=1:iters
        its += 1
        # thermalize current iterate if so desired, use final 50% of iterations to converge
        thermalize_now = anneal && (mod(i, thermalize_every) == 0) && i < (0.5iters)
        if thermalize_now
            if verbose
                println("Thermalizing with σ=$σ")
            end

            k_cur = thermalize(k_cur, σ, γ, netw.N_e, dists)
            σ = f*σ
        end

        # relaxed fixed point iterations
        J = iterate_step_mass(den, k_cur, k_base, γ, α, β)
        k_new = abs.(J).^(1/(γ+1))
        k_new = k_new./(sum(k_new.^γ)/netw.N_e).^(1/γ)

        if callback != nothing
            callback(i, k_new, thermalize_now)
        end

        Δnorm = norm(k_new - k_cur)
        oldnorm = norm(k_cur)
        Jsigns = sign(minimum(J)*maximum(J))

        if verbose && mod(i-1, show_every) == 0
            @show i, sign(minimum(J)*maximum(J)), Δnorm/oldnorm
        end

        normcheck = Δnorm < θ*oldnorm
        if (!anneal & normcheck) | (anneal & (i > 0.5iters) & normcheck)
            break
        end

        k_cur = copy(k_new)
    end

    # check that we have satisfied the convergence criteria
    # and that the gradient signs are all identical, such that our iteration
    # scheme was valid
    converged = (its < iters) && Jsigns >= 0
    # return minimizer and minimum compliance
    k_cur, objective_mass(den, k_cur .+ k_base; α=α, β=β), its, converged
end
