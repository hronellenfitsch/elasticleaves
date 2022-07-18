using Distributed
@everywhere using BSON
using Random
using Base.Filesystem

using PyCall
using PyPlot

# seed is process id.
# Random.seed!(myid())

@everywhere include("networks.jl")
@everywhere include("optimize.jl")
@everywhere include("plotting.jl")

@everywhere function make_leaf_den(N)
    pts = triang_grid(N; symmetric=true, petiole=true, elongated=false)
    netw = triangulate_mechanical_network(pts)

    n = netw.N_v

    # fix the petiole position
    fix_idx = [1, n+1, 2n+1, 2, n+2, 2n+2]

    # disallow overall rotations around the leaf axis
    den = inextensible_DEN(netw, fix_idx; remove_x_rot=true,
        remove_y_rot=false, remove_z_rot=false);
end

@everywhere function make_lilypad_den(N)
    pts = triang_grid(N; symmetric=true, petiole=false, elongated=false)
    netw = triangulate_mechanical_network(pts)

    n = netw.N_v

    # lilypad: Fix x,y,z of petiole at center and overall rotations
    center = Int64((N+1)/2)
    cor = [center - 1.0, 0., 0.] # center of rotation
    fix_idx = [center, n+center, 2n+center]

    den = inextensible_DEN(netw, fix_idx; remove_x_rot=true, remove_y_rot=true,
        remove_z_rot=true, cor_x=cor, cor_y=cor, cor_z=cor);
end

@everywhere function optimize(den, γ, k_base, anneal; show_every=50)
    netw = den.netw
    k0 = rand(netw.N_e)
    k0 = k0./(sum(k0.^γ)/netw.N_e).^(1/γ)

    if anneal != false
        iters = anneal
        do_anneal = true
    else
        do_anneal = false
        iters = 800
    end
    k_min, fg_min, its, converged = optimize_iterate(den, k0, k_base, γ;
        iters=2500, anneal=do_anneal, thermalize_every=iters, verbose=true,
        show_every=show_every)

    k_min, fg_min, converged, its, k0
end

@everywhere function plot_result(id, j, res, den::DiscreteElasticNetwork, path::String)
    # find displacements
    y, g = solve_gravity(den, res[:k_min] .+ res[:k_base])
    u = reshape(den.Φ*y, den.netw.N_v, 3)

    # plot figure
    gridspec = pyimport("matplotlib.gridspec")
    plt = pyimport("matplotlib.pyplot")

    fig = figure(figsize=(7, 3))
    gs = gridspec.GridSpec(1, 2)

    γ_plot = min(0.8res[:γ], 0.4)

    ax = fig.add_subplot(gs[1], projection="3d")
    plot_network_3d!(ax, den.netw, res[:k_min], u; γ=γ_plot)

    ax = fig.add_subplot(gs[2])
    plot_network_2d!(ax, den.netw, res[:k_min]; γ=γ_plot)
    # ax.set_zlim(-3, 3)

    fig.suptitle("N=$(res[:N]), κ_0=$(res[:k_base]), γ=$(res[:γ]), anneal=$(res[:anneal]), converged=$(res[:converged]), compliance=$(round(res[:compliance], sigdigits=3))",
        y=1.02)
    fig.tight_layout()

    plot_path = joinpath(path, "plots/", "$(id)_$(j).png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
end

@everywhere function run_optimizations(N, N_samples, make_network::Function, path::String)
    using3D()
    ioff()
    id = "$(abs(rand(Int32)))"

    combinations = []
    for k_base in [1e-6] #10.0.^LinRange(-6, 0, 4)
        for γ in [0.5]
            for anneal in [20]
                for i=1:N_samples
                    push!(combinations, [k_base, γ, anneal])
                end
            end
        end
    end

    shuffle!(combinations)

    # construct network
    den = make_network(N)

    network_path = joinpath(path, "network_$(id).bson")
    bson(network_path, Dict(:network => den))

    # Parallel compute
    @sync @distributed for (j, c) in collect(enumerate(combinations))
    # for (j, c) in collect(enumerate(combinations))
        k_base, γ, anneal = c

        k_min, fg_min, converged, its, k0 = optimize(den, γ, k_base, anneal)

        res = Dict(
            :k_min => k_min,
            :k0 => k0,
            :compliance => fg_min,
            :converged => converged,
            :iterations => its,
            :N => N,
            :k_base => k_base,
            :γ => γ,
            :anneal => anneal)

        @show res[:N], res[:k_base], res[:γ], res[:anneal]
        @show res[:converged], res[:iterations], res[:compliance]

        # plot result
        # plot_result(id, j, res, den, path)

        result_path = joinpath(path, "results_$(id)_$(j).bson")
        bson(result_path, res)
    end
end

function main(type="leaf", N=9, N_samples=1)
    @show N, N_samples
    path = "./results/$(type)/size_$(N)/"

    @show path
    mkpath(path)

    @show joinpath(path, "plots")
    mkpath(joinpath(path, "plots"))

    if type == "leaf"
        make_function = make_leaf_den
    elseif type == "lilypad"
        make_function = make_lilypad_den
    else
        println("Don't know this network type $(type)")
    end

    run_optimizations(N, N_samples, make_function, path)
end

main(ARGS[1], parse(Int64, ARGS[2]), parse(Int64, ARGS[3]))
