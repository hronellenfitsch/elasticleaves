using SparseArrays
#using Triangle
using DataStructures
using LinearAlgebra
using PyCall

tri = pyimport("matplotlib.tri")

struct MechanicalNetwork
    N_e::Int64
    N_v::Int64
    edges::Array
    node_pos::Array
    triangles::Array
    bond_vectors::Array
    bond_positions::Array
    bond_lengths::Array
    neighbor_nodes::DefaultDict
    neighbor_edges::DefaultDict
    neighbor_bond_vectors::DefaultDict
    neighbor_bond_lengths::DefaultDict
end

""" Create various variants of triangular
grids in 3d, with or without petiole, with or
without an elongated shape, or just half of a network

default setting create a leaf-like network.
"""
function triang_grid(N; symmetric=true, petiole=true, elongated=false)
    points = []

    offset = 1.0
    # petiole
    if petiole
        push!(points, [0.0, 0, 0])
        offset = 0.0
    end

    # middle row
    for i=1:N
        push!(points, [i - offset, 0.0, 0.0])
    end

    # upper and lower rows
    for i=1:Int64(floor(N/2))

        if elongated
            jmax = N
        else
            jmax = N-i
        end

        for j=1:jmax
            if symmetric
                push!(points, [j + i*0.5 - offset, i*√3/2, 0.0])
            end
            push!(points, [j + i*0.5 - offset, -i*√3/2, 0.0])
        end
    end

    hcat(points...)' |> Array
end

""" Create a Delaunay triangulation of the given points and
    return a MechanicalNetwork struct corresponding to the
    network embedded in three dimensions
"""
function triangulate_mechanical_network(pts; triangs=nothing, threshold=1.01,
                                        threshold_xy_only=false)
    # Triangle.jl sucks and is no longer maintained
    #if triangs == nothing
    #    triangs = basic_triangulation(pts, collect(1:size(pts)[1]))
    #end
    
    # use matplotlib instead
    if triangs == nothing
   	tr = tri.Triangulation(pts[:,1], pts[:,2])
    	triangs = [tr.triangles[i,:] .+ 1 for i=1:size(tr.triangles,1)]
    end

    function edge_len(e; xy_only=false)
        a, b = e
        b_vec = [pts[b,1] - pts[a,1], pts[b,2] - pts[a,2], pts[b,3] - pts[a,3]]

        if xy_only
            norm(b_vec[1:2])
        else
            norm(b_vec)
        end
    end

    edge_list = Set()
    filtered_triangs = Array{Int64}[]
    for t in triangs
        es = vcat([ [t[i], t[j]]' for i=1:length(t) for j=i+1:length(t) ]...)

        too_long = false
        for i=1:size(es, 1)
            e = sort(es[i,:])

            if edge_len(e; xy_only=threshold_xy_only) < threshold
                push!(edge_list, e)
            else
                too_long = true
            end
        end

        if !too_long
            push!(filtered_triangs, t)
        end
    end

    edge_list = collect(edge_list)
    lengths_list = [edge_len(e) for e in edge_list]
    edges = length(edge_list)

    # neighbors
    neighbor_list = DefaultDict(()->Int64[])
    bond_vectors = DefaultDict(()->Array[])
    lengths = DefaultDict(()->Float64[])
    edge_idx_list = DefaultDict(()->Int64[])
    bond_vec_list = []
    bond_pos_list = []

    for (i, (a, b)) in enumerate(edge_list)
        # construct bond vector
        b_vec = [pts[b,1] - pts[a,1], pts[b,2] - pts[a,2], pts[b,3] - pts[a,3]]
        ℓ = norm(b_vec)
        b_vec = b_vec/ℓ

        push!(bond_pos_list, [0.5(pts[b,1] + pts[a,1]), 0.5(pts[b,2] + pts[a,2]),
            0.5(pts[b,3] + pts[a,3])])
        push!(bond_vec_list, b_vec)
        push!(lengths[a], ℓ)
        push!(lengths[b], ℓ)
        push!(bond_vectors[a], b_vec)
        push!(bond_vectors[b], -b_vec)
        push!(neighbor_list[a], b)
        push!(neighbor_list[b], a)
        push!(edge_idx_list[a], i)
        push!(edge_idx_list[b], i)
    end

    MechanicalNetwork(edges, size(pts, 1), edge_list, pts, filtered_triangs,
        bond_vec_list, bond_pos_list, lengths_list,
        neighbor_list, edge_idx_list, bond_vectors, lengths)
end

"""
Create a matrix containing the squared distances between bond midpoints
of network netw
"""
function squared_bond_dists(netw::MechanicalNetwork)
    B = zeros(netw.N_e, netw.N_e)
    for i=1:netw.N_e
        for j=1:netw.N_e
            B[i,j] = sum((netw.bond_positions[i] - netw.bond_positions[j]).^2)
        end
    end
    B
end

""" Construct the equilibrium matrix of the mechanical network.
The equilibruium matrix is defined via

(Q u)_e = b_e^⊤ (u_j - u_i),

where u = [u_x, u_y, u_z] contains the nodal displacements and
b_e is the bond normal vector of edge e = (ij).
"""
function equilibrium_matrix(netw::MechanicalNetwork)
    ## Make Q matrix for inner products with bond vectors
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]

    n = netw.N_v
    for (i, ((a, b), b_vec)) in enumerate(zip(netw.edges, netw.bond_vectors))
        push!(Is, i)
        push!(Js, a)
        push!(Vs, b_vec[1])

        push!(Is, i)
        push!(Js, b)
        push!(Vs, -b_vec[1])

        push!(Is, i)
        push!(Js, n+a)
        push!(Vs, b_vec[2])

        push!(Is, i)
        push!(Js, n+b)
        push!(Vs, -b_vec[2])

        push!(Is, i)
        push!(Js, 2n+a)
        push!(Vs, b_vec[3])

        push!(Is, i)
        push!(Js, 2n+b)
        push!(Vs, -b_vec[3])
    end

    dropzeros(sparse(Is, Js, Vs, netw.N_e, 3n))
end

""" Construct the infinitesimal rotation mode around
the x axis with the given center of rotation.
"""
function x_rot_mode(netw; cor=[0.0, 0.0, 0.0])
    # add twist mode
    u = zeros(3netw.N_v)

    u[netw.N_v+1:2netw.N_v] .= -(netw.node_pos[:,3] .- cor[3])
    u[2netw.N_v+1:end] .= netw.node_pos[:,2] .- cor[2]

    u/norm(u)
end

""" Construct the infinitesimal rotation mode around
the y axis with the given center of rotation.
"""
function y_rot_mode(netw; cor=[0.0, 0.0, 0.0])
    # add twist mode
    u = zeros(3netw.N_v)

    u[1:netw.N_v] .= netw.node_pos[:,3] .- cor[3]
    u[2netw.N_v+1:end] .= -(netw.node_pos[:,1] .- cor[1])

    u/norm(u)
end

""" Construct the infinitesimal rotation mode around
the z axis with the given center of rotation.
"""
function z_rot_mode(netw; cor=[0.0, 0.0, 0.0])
    # add twist mode
    # add twist mode
    u = zeros(3netw.N_v)

    u[1:netw.N_v] .= -(netw.node_pos[:,2] .- cor[2])
    u[netw.N_v+1:2netw.N_v] .= netw.node_pos[:,1] .- cor[1]

    u/norm(u)
end

""" Models discrete elastic networks
"""
struct DiscreteElasticNetwork
    netw::MechanicalNetwork
    Ds::DefaultDict
    Φ::AbstractArray # projects on allowed degrees of freedom
    dofs::Int64 # number of allowed degrees of freedom
end

# construct D matrices to compute cross produces (b × Δu)/ℓ
# returns a dict with lists of matices, one for each neighbor
function D_matrices(netw::MechanicalNetwork, Φ=I)
    D_list = DefaultDict(()->Any[])
    n = netw.N_v

    for i=1:n
        for (neigh, b, ℓ) in zip(netw.neighbor_nodes[i],
            netw.neighbor_bond_vectors[i], netw.neighbor_bond_lengths[i])
            Is = Int64[]
            Js = Int64[]
            V = Float64[]

            # x component of cross product
            push!(Is, 1)
            push!(Is, 1)
            push!(Is, 1)
            push!(Is, 1)
            push!(Js, i+2n) # u_iz
            push!(Js, neigh+2n) # u_jz
            push!(Js, i+n) # u_iy
            push!(Js, neigh+n) #u_jy
            push!(V, -b[2]/ℓ)
            push!(V, b[2]/ℓ)
            push!(V, b[3]/ℓ)
            push!(V, -b[3]/ℓ)

            # y component
            push!(Is, 2)
            push!(Is, 2)
            push!(Is, 2)
            push!(Is, 2)
            push!(Js, i+2n) #u_iz
            push!(Js, neigh+2n) #u_jz
            push!(Js, i) # u_ix
            push!(Js, neigh) # u_jx
            push!(V, b[1]/ℓ)
            push!(V, -b[1]/ℓ)
            push!(V, -b[3]/ℓ)
            push!(V, b[3]/ℓ)

            # z component
            push!(Is, 3)
            push!(Is, 3)
            push!(Is, 3)
            push!(Is, 3)
            push!(Js, i) #u_ix
            push!(Js, neigh) #u_jx
            push!(Js, i+n) #u_iy
            push!(Js, neigh+n) #u_jy
            push!(V, b[2]/ℓ)
            push!(V, -b[2]/ℓ)
            push!(V, -b[1]/ℓ)
            push!(V, b[1]/ℓ)

            Db = sparse(Is, Js, V, 3, 3n)

            push!(D_list[i], Db*Φ)
        end
    end

    D_list
end

""" Helper function to create an inextensible DEN
with the displacement indices fix set to zero
"""
function inextensible_DEN(netw::MechanicalNetwork, fix::Array;
    remove_x_rot=true, remove_y_rot=true, remove_z_rot=true,
    cor_x=[0., 0., 0.], cor_y=[0., 0., 0.], cor_z=[0., 0., 0.])
    n = netw.N_v

    # inextensibility
    Q = equilibrium_matrix(netw) |> Array

    # fixed nodes
    fixed_displacements = zeros(length(fix), 3n)
    for (i, i_fix) in enumerate(fix)
        fixed_displacements[i,i_fix] = 1.0
    end

    # constraint matrix
    Q = vcat(Q, fixed_displacements)

    # remove rotational degrees of freedom around the origin
    if remove_x_rot
        u_rot = x_rot_mode(netw; cor=cor_x)
        Q = vcat(Q, u_rot')
    end

    if remove_y_rot
        u_rot = y_rot_mode(netw; cor=cor_y)
        Q = vcat(Q, u_rot')
    end

    if remove_z_rot
        u_rot = z_rot_mode(netw; cor=cor_z)
        Q = vcat(Q, u_rot')
    end

    # allowed degrees of freedom
    Φ = nullspace(Q)
    dofs = size(Φ, 2)

    Ds = D_matrices(netw, Φ)
    DiscreteElasticNetwork(netw, Ds, Φ, dofs)
end

""" Construct the DEN bending Hessian given elastic
constants k.
"""
function network_hessian(den::DiscreteElasticNetwork, k::Array)
    # construct the Hessian for bending constants k
    netw = den.netw
    n = netw.N_v
    dofs = den.dofs

    # can be spzeros
    H = zeros(dofs, dofs)
    for i=1:n
        k_neigh = k[netw.neighbor_edges[i]]

        # compute C_i
        C_i = sum(kj*(I - b*b') for (kj, b) in zip(k_neigh,
            netw.neighbor_bond_vectors[i]))

        # compute D_i
        D_i = sum(k_neigh.*den.Ds[i])

        # not ideal but works and keeps things sparse if sparse(inv(C_i))
        CinvD_i = pinv(C_i) * D_i

        # compute Hessian
        D_sqr = sum(kj*(D'*D) for (kj, D) in zip(k_neigh, den.Ds[i]))
        D_Cinv_D = D_i'*CinvD_i

#         push!(H_list, D_sqr - D_Cinv_D)
        H += D_sqr - D_Cinv_D
    end

    # Hessian matrix for the model
    Hermitian(H)
end

""" Compute the matrix F⃗_{ij} of nodal forces for the given DBN
with bending stiffnesses k and projected displacements y.

TODO: This redoes some of the calculations necessary to obtain y,
so it could be made more efficient
"""
function network_edge_forces(den::DiscreteElasticNetwork, k::Array, y::Array)
    # construct the Hessian for bending constants k
    netw = den.netw
    n = netw.N_v
    dofs = den.dofs

    # can be spzeros
    F = zeros(netw.N_v, netw.N_v, 3)

    for i=1:n
        k_neigh = k[netw.neighbor_edges[i]]

        # compute C_i
        C_i = sum(kj*(I - b*b') for (kj, b) in zip(k_neigh,
            netw.neighbor_bond_vectors[i]))

        # compute D_i
        D_i = sum(k_neigh.*den.Ds[i])

        # solve C_i n = D_i u
        rhs = D_i*y
        n = pinv(C_i) * rhs

        # calculate forces
        for (kn, j, b, Db) in zip(k[netw.neighbor_edges[i]], netw.neighbor_nodes[i],
                netw.neighbor_bond_vectors[i], den.Ds[i])
            C_b = I - b*b'

            Fvec = -kn*cross(b, C_b*n - Db*y)

            F[i,j,:] .= Fvec
        end
    end

    # Force matrix
    F
end

""" Compute the sin²α_{ij} + sin²α_{ji} for each edge (ij) given the bending stiffnesses
k and the corresponding projected displacements y.

TODO: This redoes some of the calculations necessary to obtain y,
so it could be made more efficient
"""
function network_edge_angles(den::DiscreteElasticNetwork, k::Array, y::Array)
    # construct the Hessian for bending constants k
    netw = den.netw
    n = netw.N_v
    dofs = den.dofs

    # can be spzeros
    sin_sqrs = zeros(netw.N_e)

    for i=1:n
        k_neigh = k[netw.neighbor_edges[i]]

        # compute C_i
        C_i = sum(kj*(I - b*b') for (kj, b) in zip(k_neigh,
            netw.neighbor_bond_vectors[i]))

        # compute D_i
        D_i = sum(k_neigh.*den.Ds[i])

        # solve C_i n = D_i u
        rhs = D_i*y
        n = pinv(C_i) * rhs

        # calculate angles for each edge
        for (kn, b_neigh, j, b, Db) in zip(k[netw.neighbor_edges[i]], netw.neighbor_edges[i],
                netw.neighbor_nodes[i], netw.neighbor_bond_vectors[i], den.Ds[i])
            sin_sqrs[b_neigh] += sum((n - (b'*n)*b - Db*y).^2)
        end
    end

    # angles
    sin_sqrs
end
