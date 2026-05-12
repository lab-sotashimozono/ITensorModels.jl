using ITensors: SiteType, @SiteType_str

"""
    LongRangeXY1D(; J=1.0, α=3.0, site=SiteType("S=1/2"))

1D power-law XY chain:

    H = Σ_{i < j} [J / |i - j|^α] (Sˣᵢ Sˣⱼ + Sʸᵢ Sʸⱼ)

every pair is coupled with strength decaying as `1/r^α`. Spin model
relevant to 1D Rydberg / trapped-ion quantum simulators (dipole-dipole
`α = 3`, van der Waals `α = 6`).

`α → ∞` reduces to the nearest-neighbor XX chain. For `α < 1` the
model is in the "strong long-range" regime with extensive GS energy
density.

Overrides [`local_ham_terms`](@ref) to emit all O(N²) pair couplings
directly. Does not fit the 1D `ModulatedModel` nearest-neighbor wrapper.
"""
Base.@kwdef struct LongRangeXY1D <: AbstractLatticeModel
    J::Float64 = 1.0
    α::Float64 = 3.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::LongRangeXY1D) = m.site

function _xy_bond(J::Real, ::SiteType"S=1/2", i::Int, j::Int)
    H = OpSum()
    H += J, "Sx", i, "Sx", j
    H += J, "Sy", i, "Sy", j
    return H
end

bond_term(m::LongRangeXY1D, i::Int, j::Int) = _xy_bond(m.J, m.site, i, j)
bond_coupling_term(m::LongRangeXY1D, i::Int, j::Int) = _xy_bond(m.J, m.site, i, j)
onsite_term(::LongRangeXY1D, ::Int) = OpSum()

function onsite_observable_op(::LongRangeXY1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("LongRangeXY1D: unsupported onsite observable $name")
end

function local_ham_terms(m::LongRangeXY1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for i in 1:(N - 1), j in (i + 1):N
        Jij = m.J / abs(i - j)^m.α
        push!(terms, _xy_bond(Jij, m.site, phys[i], phys[j]))
    end
    return terms
end
