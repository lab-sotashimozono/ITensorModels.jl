using ITensors: SiteType, @SiteType_str

"""
    AKLT1D(; J=1.0, site=SiteType("S=1"))

1D Affleck–Kennedy–Lieb–Tasaki (AKLT) chain on spin-1 sites:

    H = J Σ_i [ Sᵢ · Sᵢ₊₁ + (1/3) (Sᵢ · Sᵢ₊₁)² ]

The biquadratic-to-bilinear ratio `1/3` is the AKLT point of the
bilinear-biquadratic (BLBQ) family — the ground state is an exact
matrix-product state, the bulk gap is finite (≈ 0.350 J), and the
open chain hosts free `S = 1/2` edge modes that give rise to a
4-fold degenerate ground space (the Haldane SPT phase).

References:
- Affleck, Kennedy, Lieb, Tasaki, PRL 59, 799 (1987).
- Affleck, Kennedy, Lieb, Tasaki, Comm. Math. Phys. 115, 477 (1988).
"""
Base.@kwdef struct AKLT1D <: AbstractLatticeModel
    J::Float64 = 1.0
    site::SiteType = SiteType("S=1")
end

site_type(m::AKLT1D) = m.site

const _AKLT_OPS = ("Sx", "Sy", "Sz")

function _aklt_bond(m::AKLT1D, i::Int, j::Int)
    H = OpSum()
    # Bilinear: J * S_i · S_j.
    for α in _AKLT_OPS
        H += m.J, α, i, α, j
    end
    # Biquadratic: (J/3) * (S_i · S_j)^2 = (J/3) Σ_{α,β} S^α_i S^β_i S^α_j S^β_j.
    for α in _AKLT_OPS, β in _AKLT_OPS
        H += m.J / 3, α, i, β, i, α, j, β, j
    end
    return H
end

bond_term(m::AKLT1D, i::Int, j::Int) = _aklt_bond(m, i, j)
bond_coupling_term(m::AKLT1D, i::Int, j::Int) = _aklt_bond(m, i, j)
onsite_term(::AKLT1D, ::Int) = OpSum()

function onsite_observable_op(m::AKLT1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("AKLT1D: unsupported onsite observable $name")
end
