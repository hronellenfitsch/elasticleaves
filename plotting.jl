using PyCall
using PyPlot
using3D()

tri = pyimport("matplotlib.tri")

# plotting code
function plot_network_2d!(ax, netw, k; node_labels=false, γ=0.35, alpha=1.0)
    if node_labels
        for i=1:netw.N_v
            ax.text(netw.node_pos[i,1]+0.15, netw.node_pos[i,2]+0.05, s="$i")
        end
    end

    for (kj, (a, b)) in zip(k, netw.edges)
        x1 = netw.node_pos[a,1]
        x2 = netw.node_pos[b,1]

        y1 = netw.node_pos[a,2]
        y2 = netw.node_pos[b,2]
        ax.plot([x1, x2], [y1, y2], color="k", lw=kj.^γ, alpha=alpha, zorder=-100)
    end
end

function average_over_triangles(netw::MechanicalNetwork, s)
    """ take the scalar data s defined on the nodes and
    average it over the triangles in the triangulation of netw.
    """
    [(s[a] + s[b] + s[c])/3 for (a, b, c) in netw.triangles]
end

function plot_network_3d!(ax, netw, k, u; c=u[:,3], plot_surface=true,
    cmap="viridis", alpha=1.0, alpha_surf=0.5, viewx=30, viewy=-45, γ=0.35, normalize_colors=false,
    vlimits=nothing, zorder_surf=-100, zorder_netw=Inf)
    surf = nothing
    if plot_surface
        surf = ax.plot_trisurf(tri.Triangulation(netw.node_pos[:,1] .+ u[:,1],
            netw.node_pos[:,2] .+ u[:,2], [t .- 1 for t in netw.triangles]),
            netw.node_pos[:,3] .+ u[:,3], alpha=alpha_surf, cmap=cmap, zorder=zorder_surf)

        colors = average_over_triangles(netw, c)

        if normalize_colors
            colors = colors/maximum(colors)
        end

        surf.set_array(colors)
        vmin = minimum(colors)
        vmax = maximum(colors)

        if vlimits != nothing
            vmin, vmax = vlimits
        end
        surf.set_clim(vmin, vmax)
    end

    edges = []
    for (kj, (a, b)) in zip(k, netw.edges)
        x1 = netw.node_pos[a,1] .+ u[a,1]
        x2 = netw.node_pos[b,1] .+ u[b,1]

        y1 = netw.node_pos[a,2] .+ u[a,2]
        y2 = netw.node_pos[b,2] .+ u[b,2]

        z1 = netw.node_pos[a,3] .+ u[a,3]
        z2 = netw.node_pos[b,3] .+ u[b,3]
        e = ax.plot([x1, x2], [y1, y2], [z1, z2], color="k", lw=kj.^γ, zorder=zorder_netw,
            alpha=alpha)

        push!(edges, e)
    end

    ax.view_init(viewx, viewy)

    surf, edges
end
