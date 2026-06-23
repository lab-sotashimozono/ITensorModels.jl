"""
    Heisenberg1D(; J=1.0, site=SiteType("S=1/2"))

Convenience wrapper for the isotropic point of [`XXZ1D`](@ref):

    H = J Σ [ Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁ ]

Matches `QAtlas.Heisenberg1D`. Structurally a thin alias around
`XXZ1D(J, Δ=1, site)` that reuses its `bond_term`.
"""
Base.@kwdef struct Heisenberg1D <: AbstractLatticeModel
    J::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::Heisenberg1D) = m.site

function bond_term(m::Heisenberg1D, i::Int, j::Int)
    return bond_term(XXZ1D(; J=m.J, Δ=1.0, site=m.site), i, j)
end

function onsite_observable_op(m::Heisenberg1D, name::Symbol)
    return onsite_observable_op(XXZ1D(; J=m.J, Δ=1.0, site=m.site), name)
end

# ---------------------------------------------------------------------
# Split protocol: delegate to XXZ1D at Δ=1 — same alias pattern as
# `bond_term(::Heisenberg1D)`.
# ---------------------------------------------------------------------

function bond_coupling_term(m::Heisenberg1D, i::Int, j::Int)
    return bond_coupling_term(XXZ1D(; J=m.J, Δ=1.0, site=m.site), i, j)
end

onsite_term(::Heisenberg1D, ::Int) = OpSum()
