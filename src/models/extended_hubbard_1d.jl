using ITensors: SiteType, @SiteType_str

"""
    ExtendedHubbard1D(; t=1.0, U=4.0, V=1.0, μ=0.0, site=SiteType("Electron"))

1D extended (t-U-V) Hubbard model:

    H = -t Σ_⟨ij⟩σ (c†_iσ c_jσ + h.c.)
        + U Σ_i n_{i↑} n_{i↓}
        + V Σ_⟨ij⟩ n_i n_j
        - μ Σ_i n_i

on `SiteType("Electron")` (auto Jordan-Wigner via OpSum). The
nearest-neighbor density-density `V` term stabilizes charge-density-wave
(CDW) order, giving a richer 1D phase diagram (Lin & Hirsch 1986;
Nakamura 2000) than the pure-Hubbard line: SDW / CDW / phase-separation
boundaries depending on `U / V`.
"""
Base.@kwdef struct ExtendedHubbard1D <: AbstractLatticeModel
    t::Float64 = 1.0
    U::Float64 = 4.0
    V::Float64 = 1.0
    μ::Float64 = 0.0
    site::SiteType = SiteType("Electron")
end

site_type(m::ExtendedHubbard1D) = m.site

function bond_term(m::ExtendedHubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdagup", i, "Cup", j
    H += -m.t, "Cdagup", j, "Cup", i
    H += -m.t, "Cdagdn", i, "Cdn", j
    H += -m.t, "Cdagdn", j, "Cdn", i
    H += m.V, "Ntot", i, "Ntot", j
    H += (m.U / 2), "Nupdn", i
    H += (m.U / 2), "Nupdn", j
    H += -(m.μ / 2), "Ntot", i
    H += -(m.μ / 2), "Ntot", j
    return H
end

function boundary_patch(m::ExtendedHubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 2), "Nupdn", k
    H += -(m.μ / 2), "Ntot", k
    return H
end

function bond_coupling_term(m::ExtendedHubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdagup", i, "Cup", j
    H += -m.t, "Cdagup", j, "Cup", i
    H += -m.t, "Cdagdn", i, "Cdn", j
    H += -m.t, "Cdagdn", j, "Cdn", i
    H += m.V, "Ntot", i, "Ntot", j
    return H
end

function onsite_term(m::ExtendedHubbard1D, k::Int)
    H = OpSum()
    H += m.U, "Nupdn", k
    H += -m.μ, "Ntot", k
    return H
end

function onsite_observable_op(::ExtendedHubbard1D, name::Symbol)
    name === :nup && return "Nup"
    name === :ndn && return "Ndn"
    name === :ntot && return "Ntot"
    name === :nupdn && return "Nupdn"
    name === :sz && return "Sz"
    return error("ExtendedHubbard1D: unsupported onsite observable $name")
end
