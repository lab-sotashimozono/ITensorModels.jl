using ITensors: SiteType, @SiteType_str

"""
    TightBinding1D(; t=1.0, μ=0.0, site=SiteType("Fermion"))

1D spinless-fermion tight-binding chain

    H = -t Σ_i (c†_i c_{i+1} + c†_{i+1} c_i)  +  μ Σ_i n_i

on `SiteType("Fermion")` sites. Jordan-Wigner strings between non-
adjacent operators are inserted automatically by ITensors' OpSum →
MPO conversion.

This is the canonical free-fermion model: at half-filling (μ = 0)
the open chain has analytic single-particle eigenvalues
`ε_k = -2t cos(kπ/(N+1))` for `k ∈ {1, ..., N}`, and the ground-
state energy is the sum of the negative eigenvalues. Equivalent to
the XX chain under Jordan-Wigner transformation.

QAtlas currently exposes 2D tight-binding only (Honeycomb / Kagome /
Lieb / Triangular); a direct 1D bridge is intentionally deferred.
"""
Base.@kwdef struct TightBinding1D <: AbstractLatticeModel
    t::Float64 = 1.0
    μ::Float64 = 0.0
    site::SiteType = SiteType("Fermion")
end

site_type(m::TightBinding1D) = m.site

function bond_term(m::TightBinding1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdag", i, "C", j
    H += -m.t, "Cdag", j, "C", i
    H += (m.μ / 2), "N", i
    H += (m.μ / 2), "N", j
    return H
end

function boundary_patch(m::TightBinding1D, k::Int)
    H = OpSum()
    H += (m.μ / 2), "N", k
    return H
end

function onsite_observable_op(m::TightBinding1D, name::Symbol)
    name === :n && return "N"
    name === :c && return "C"
    name === :cdag && return "Cdag"
    return error(
        "TightBinding1D: unsupported onsite observable $name on site $(site_type(m))"
    )
end

function bond_coupling_term(m::TightBinding1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdag", i, "C", j
    H += -m.t, "Cdag", j, "C", i
    return H
end

function onsite_term(m::TightBinding1D, k::Int)
    H = OpSum()
    H += m.μ, "N", k
    return H
end
