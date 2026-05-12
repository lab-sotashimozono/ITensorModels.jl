using ITensors: SiteType, @SiteType_str

"""
    S1AnisotropicD1D(; J=1.0, D=0.0, site=SiteType("S=1"))

S=1 Heisenberg chain with single-ion (uniaxial) anisotropy:

    H = J Σ_i Sᵢ · Sᵢ₊₁ + D Σ_i (Sᶻᵢ)²

Phase diagram (Schulz 1986; Tzeng 2008):
- `D = 0`: gapped Haldane phase.
- Large `D > 0`: "large-D" trivial paramagnetic phase (gap reopens).
- Large `D < 0` (easy-axis): Ising-like Néel order.

The Haldane → large-D transition is Gaussian-critical at `D/J ≈ 0.97`.
Useful benchmark for SPT-phase numerics.
"""
Base.@kwdef struct S1AnisotropicD1D <: AbstractLatticeModel
    J::Float64 = 1.0
    D::Float64 = 0.0
    site::SiteType = SiteType("S=1")
end

site_type(m::S1AnisotropicD1D) = m.site

function bond_term(m::S1AnisotropicD1D, i::Int, j::Int)
    H = OpSum()
    H += m.J, "Sx", i, "Sx", j
    H += m.J, "Sy", i, "Sy", j
    H += m.J, "Sz", i, "Sz", j
    H += (m.D / 2), "Sz", i, "Sz", i
    H += (m.D / 2), "Sz", j, "Sz", j
    return H
end

function boundary_patch(m::S1AnisotropicD1D, k::Int)
    H = OpSum()
    H += (m.D / 2), "Sz", k, "Sz", k
    return H
end

function bond_coupling_term(m::S1AnisotropicD1D, i::Int, j::Int)
    H = OpSum()
    H += m.J, "Sx", i, "Sx", j
    H += m.J, "Sy", i, "Sy", j
    H += m.J, "Sz", i, "Sz", j
    return H
end

function onsite_term(m::S1AnisotropicD1D, k::Int)
    H = OpSum()
    H += m.D, "Sz", k, "Sz", k
    return H
end

function onsite_observable_op(::S1AnisotropicD1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("S1AnisotropicD1D: unsupported onsite observable $name")
end
