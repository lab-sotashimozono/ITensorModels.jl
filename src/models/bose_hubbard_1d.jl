using ITensors: SiteType, @SiteType_str

"""
    BoseHubbard1D(; t=1.0, U=2.0, μ=0.0, site=SiteType("Boson"))

1D Bose-Hubbard model

    H = -t Σ_i (b†_i b_{i+1} + b†_{i+1} b_i)
      + (U/2) Σ_i n_i (n_i - 1)
      + μ Σ_i n_i

on `SiteType("Boson")` sites. The local Hilbert space dimension is
set by the caller via `siteinds("Boson", N; dim=k)` (operators `A` /
`Adag` / `N` work at any truncation; the model is dim-agnostic).

The 1D Bose-Hubbard model exhibits a Mott-superfluid quantum phase
transition at integer fillings (Fisher, Weichman, Grinstein, Fisher,
PRB 40, 546 (1989); Greiner et al. (2002) optical-lattice setup).
"""
Base.@kwdef struct BoseHubbard1D <: AbstractLatticeModel
    t::Float64 = 1.0
    U::Float64 = 2.0
    μ::Float64 = 0.0
    site::SiteType = SiteType("Boson")
end

site_type(m::BoseHubbard1D) = m.site

function bond_term(m::BoseHubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Adag", i, "A", j
    H += -m.t, "Adag", j, "A", i
    # Half-weight on-site U/2 n(n-1) + μ n on each endpoint.
    H += (m.U / 4), "N", i, "N", i
    H += (-m.U / 4 + m.μ / 2), "N", i
    H += (m.U / 4), "N", j, "N", j
    H += (-m.U / 4 + m.μ / 2), "N", j
    return H
end

function boundary_patch(m::BoseHubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 4), "N", k, "N", k
    H += (-m.U / 4 + m.μ / 2), "N", k
    return H
end

function onsite_observable_op(m::BoseHubbard1D, name::Symbol)
    name === :n && return "N"
    name === :a && return "A"
    name === :adag && return "Adag"
    return error("BoseHubbard1D: unsupported onsite observable $name")
end

function bond_coupling_term(m::BoseHubbard1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Adag", i, "A", j
    H += -m.t, "Adag", j, "A", i
    return H
end

function onsite_term(m::BoseHubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 2), "N", k, "N", k
    H += (-m.U / 2 + m.μ), "N", k
    return H
end
