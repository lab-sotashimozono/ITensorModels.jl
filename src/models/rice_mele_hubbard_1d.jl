using ITensors: SiteType, @SiteType_str

"""
    RiceMeleHubbard1D(; v=1.0, w=1.0, Δ=0.0, U=0.0, site=SiteType("Electron"))

Rice-Mele chain with on-site Hubbard repulsion, for spinful fermions:

    H = -v Σ_{i odd,σ}  (c†_iσ c_{i+1,σ} + h.c.)
        -w Σ_{i even,σ} (c†_iσ c_{i+1,σ} + h.c.)
        + Δ Σ_i (-1)^{i+1} n_i
        + U Σ_i n_{i↑} n_{i↓}

Dimerised hopping (`v` intracell, `w` intercell) plus a staggered potential `Δ`
that opens a gap the SSH chain's chiral symmetry forbids: `√((|v|−|w|)² + Δ²)`,
which never closes. At `U = 0` the two spin species decouple and each fills the
lower band at half filling, so the energy is twice the spinless Rice-Mele value —
`test/base/test_rice_mele_hubbard_1d_vs_qatlas.jl` uses that against QAtlas.

`Δ` is the full sublattice offset, not half of it: site 1 sits at `+Δ`.
"""
Base.@kwdef struct RiceMeleHubbard1D <: AbstractLatticeModel
    v::Float64 = 1.0
    w::Float64 = 1.0
    Δ::Float64 = 0.0
    U::Float64 = 0.0
    site::SiteType = SiteType("Electron")
end

site_type(m::RiceMeleHubbard1D) = m.site

_hop(m::RiceMeleHubbard1D, i::Int) = isodd(i) ? m.v : m.w
_stagger(m::RiceMeleHubbard1D, k::Int) = isodd(k) ? m.Δ : -m.Δ

function bond_coupling_term(m::RiceMeleHubbard1D, i::Int, j::Int)
    t = _hop(m, i)
    H = OpSum()
    H += -t, "Cdagup", i, "Cup", j
    H += -t, "Cdagup", j, "Cup", i
    H += -t, "Cdagdn", i, "Cdn", j
    H += -t, "Cdagdn", j, "Cdn", i
    return H
end

function onsite_term(m::RiceMeleHubbard1D, k::Int)
    H = OpSum()
    H += m.U, "Nupdn", k
    H += _stagger(m, k), "Ntot", k
    return H
end

function bond_term(m::RiceMeleHubbard1D, i::Int, j::Int)
    H = bond_coupling_term(m, i, j)
    H += (m.U / 2), "Nupdn", i
    H += (m.U / 2), "Nupdn", j
    H += _stagger(m, i) / 2, "Ntot", i
    H += _stagger(m, j) / 2, "Ntot", j
    return H
end

function boundary_patch(m::RiceMeleHubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 2), "Nupdn", k
    H += _stagger(m, k) / 2, "Ntot", k
    return H
end

function onsite_observable_op(::RiceMeleHubbard1D, name::Symbol)
    name === :n && return "Ntot"
    name === :nupdn && return "Nupdn"
    return error("RiceMeleHubbard1D: unsupported onsite observable $name")
end
