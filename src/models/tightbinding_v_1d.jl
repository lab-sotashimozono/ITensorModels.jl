using ITensors: SiteType, @SiteType_str

"""
    TightBindingV1D(; t=1.0, V=1.0, μ=0.0, site=SiteType("Fermion"))

1D spinless-fermion t-V model

    H = -t Σ_i (c†_i c_{i+1} + c†_{i+1} c_i)
      + V Σ_i n_i n_{i+1}
      + μ Σ_i n_i

on `SiteType("Fermion")` sites. The `V` term is a nearest-neighbor
density-density interaction; at half-filling the model is exactly
solvable by Bethe ansatz and equivalent to the XXZ chain at
`Δ = V / (2t)` under Jordan-Wigner.

`V = 0` reduces to [`TightBinding1D`](@ref). `V > 2t` at half-filling
opens a charge-density-wave gap.
"""
Base.@kwdef struct TightBindingV1D <: AbstractLatticeModel
    t::Float64 = 1.0
    V::Float64 = 1.0
    μ::Float64 = 0.0
    site::SiteType = SiteType("Fermion")
end

site_type(m::TightBindingV1D) = m.site

function bond_term(m::TightBindingV1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdag", i, "C", j
    H += -m.t, "Cdag", j, "C", i
    H += m.V, "N", i, "N", j
    H += (m.μ / 2), "N", i
    H += (m.μ / 2), "N", j
    return H
end

function boundary_patch(m::TightBindingV1D, k::Int)
    H = OpSum()
    H += (m.μ / 2), "N", k
    return H
end

function onsite_observable_op(m::TightBindingV1D, name::Symbol)
    name === :n && return "N"
    name === :c && return "C"
    name === :cdag && return "Cdag"
    return error("TightBindingV1D: unsupported onsite observable $name")
end

function bond_coupling_term(m::TightBindingV1D, i::Int, j::Int)
    H = OpSum()
    H += -m.t, "Cdag", i, "C", j
    H += -m.t, "Cdag", j, "C", i
    H += m.V, "N", i, "N", j
    return H
end

function onsite_term(m::TightBindingV1D, k::Int)
    H = OpSum()
    H += m.μ, "N", k
    return H
end
