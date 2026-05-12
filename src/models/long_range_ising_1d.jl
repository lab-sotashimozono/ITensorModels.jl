using ITensors: SiteType, @SiteType_str

"""
    LongRangeIsing1D(; J=1.0, α=3.0, h=0.0, site=SiteType("Qubit"))

1D power-law transverse-field Ising model:

    H = -Σ_{i<j} [J / |i-j|^α] Z_i Z_j  -  h Σ_i X_i

on `SiteType("Qubit")`. Power-law ZZ couplings (positive `J` =
ferromagnetic). `α → ∞` recovers the nearest-neighbor TFIM; `α < 2`
enters the strong-long-range regime.

Direct model for trapped-ion arrays (Britton et al., Nature 2012)
and 1D Rydberg dipole-dipole simulators (`α = 3`).
"""
Base.@kwdef struct LongRangeIsing1D <: AbstractLatticeModel
    J::Float64 = 1.0
    α::Float64 = 3.0
    h::Float64 = 0.0
    site::SiteType = SiteType("Qubit")
end

site_type(m::LongRangeIsing1D) = m.site

bond_term(::LongRangeIsing1D, ::Int, ::Int) = OpSum()
bond_coupling_term(::LongRangeIsing1D, ::Int, ::Int) = OpSum()

function onsite_term(m::LongRangeIsing1D, k::Int)
    H = OpSum()
    H += -m.h, "X", k
    return H
end

function onsite_observable_op(::LongRangeIsing1D, name::Symbol)
    name === :x && return "X"
    name === :y && return "Y"
    name === :z && return "Z"
    return error("LongRangeIsing1D: unsupported onsite observable $name")
end

function local_ham_terms(m::LongRangeIsing1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for i in 1:(N - 1), j in (i + 1):N
        Jij = m.J / abs(i - j)^m.α
        H = OpSum()
        H += -Jij, "Z", phys[i], "Z", phys[j]
        push!(terms, H)
    end
    if !iszero(m.h)
        for k in 1:N
            H = OpSum()
            H += -m.h, "X", phys[k]
            push!(terms, H)
        end
    end
    return terms
end
