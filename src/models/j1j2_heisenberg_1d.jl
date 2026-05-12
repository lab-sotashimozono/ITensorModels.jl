using ITensors: SiteType, @SiteType_str

"""
    J1J2Heisenberg1D(; J1=1.0, J2=0.5, site=SiteType("S=1/2"))

1D spin-½ J1-J2 Heisenberg chain (frustrated):

    H = J1 Σ_i Sᵢ · Sᵢ₊₁  +  J2 Σ_i Sᵢ · Sᵢ₊₂

BKT transition at `J2/J1 ≈ 0.241` between a gapless Tomonaga-Luttinger
liquid and a dimerized gapped phase. At the Majumdar–Ghosh point
`J2 = J1/2` the ground state is the exact dimer product (Majumdar,
Ghosh, J. Math. Phys. 10, 1388 (1969)).

The model overrides [`local_ham_terms`](@ref) directly to emit both
NN and NNN bond terms; the standard single-bond `bond_term(model, i, j)`
returns only the NN piece. Does not fit the 1D nearest-neighbor
`ModulatedModel` wrapper because of the NNN coupling.
"""
Base.@kwdef struct J1J2Heisenberg1D <: AbstractLatticeModel
    J1::Float64 = 1.0
    J2::Float64 = 0.5
    site::SiteType = SiteType("S=1/2")
end

site_type(m::J1J2Heisenberg1D) = m.site

function _heis_bond(J::Real, ::SiteType"S=1/2", i::Int, j::Int)
    H = OpSum()
    H += J, "Sx", i, "Sx", j
    H += J, "Sy", i, "Sy", j
    H += J, "Sz", i, "Sz", j
    return H
end

bond_term(m::J1J2Heisenberg1D, i::Int, j::Int) = _heis_bond(m.J1, m.site, i, j)
bond_coupling_term(m::J1J2Heisenberg1D, i::Int, j::Int) = _heis_bond(m.J1, m.site, i, j)
onsite_term(::J1J2Heisenberg1D, ::Int) = OpSum()

function onsite_observable_op(::J1J2Heisenberg1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("J1J2Heisenberg1D: unsupported onsite observable $name")
end

function local_ham_terms(m::J1J2Heisenberg1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for k in 1:(N - 1)
        push!(terms, _heis_bond(m.J1, m.site, phys[k], phys[k + 1]))
    end
    for k in 1:(N - 2)
        push!(terms, _heis_bond(m.J2, m.site, phys[k], phys[k + 2]))
    end
    return terms
end
