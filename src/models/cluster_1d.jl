using ITensors: SiteType, @SiteType_str

"""
    Cluster1D(; h=1.0, site=SiteType("Qubit"))

1D cluster Hamiltonian (textbook ℤ₂ × ℤ₂ SPT model):

    H = -h Σ_i Z_{i-1} X_i Z_{i+1}

on `SiteType("Qubit")`. Ground state is the 1D cluster state — a
canonical resource for measurement-based quantum computing
(Raussendorf and Briegel, PRL 86, 5188 (2001)). The model exhibits
a symmetry-protected topological (SPT) phase under ℤ₂ × ℤ₂ generated
by the products of `X` on even / odd sublattices; open boundaries
host free edge `Z` operators (4-fold GS degeneracy).

Overrides [`local_ham_terms`](@ref) since the elementary interaction
is 3-site. OBC: end sites have no cluster term.
"""
Base.@kwdef struct Cluster1D <: AbstractLatticeModel
    h::Float64 = 1.0
    site::SiteType = SiteType("Qubit")
end

site_type(m::Cluster1D) = m.site

bond_term(::Cluster1D, ::Int, ::Int) = OpSum()
bond_coupling_term(::Cluster1D, ::Int, ::Int) = OpSum()
onsite_term(::Cluster1D, ::Int) = OpSum()

function onsite_observable_op(::Cluster1D, name::Symbol)
    name === :x && return "X"
    name === :y && return "Y"
    name === :z && return "Z"
    return error("Cluster1D: unsupported onsite observable $name")
end

function local_ham_terms(m::Cluster1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for k in 2:(N - 1)
        H = OpSum()
        H += -m.h, "Z", phys[k - 1], "X", phys[k], "Z", phys[k + 1]
        push!(terms, H)
    end
    return terms
end
