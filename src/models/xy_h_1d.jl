using ITensors: SiteType, @SiteType_str

"""
    XYh1D(; J=1.0, γ=0.0, h=0.0, site=SiteType("S=1/2"))

1D anisotropic XY model in a transverse field (the canonical
Lieb-Schultz-Mattis chain):

    H = -J Σ_i [ (1+γ) S^x_i S^x_{i+1} + (1-γ) S^y_i S^y_{i+1} ]
        - h Σ_i S^z_i

on `SiteType("S=1/2")`. Special cases:

- `γ = 0`: isotropic XX chain (free-fermion line via Jordan-Wigner; the
  field `h` becomes a chemical potential on the JW fermions).
- `γ = 1`: equivalent to the transverse-field Ising chain after a
  global `π/2` spin rotation. Note that with `S = 1/2` operators
  (`Sˣ = σˣ / 2`), the coupling reads `-J SˣSˣ = -(J/4) σˣσˣ`, so the
  TFIM coupling in Pauli-operator conventions is `J/4`, not `J`.
- `γ ≠ 0`: quantum phase transition at `|h| / J = 1` (Ising universality
  class; the transition persists at `γ = 1`).

Exactly solvable by JW + Bogoliubov (Lieb, Schultz, Mattis 1961); useful
benchmark for DMRG and quench dynamics.
"""
Base.@kwdef struct XYh1D <: AbstractLatticeModel
    J::Float64 = 1.0
    γ::Float64 = 0.0
    h::Float64 = 0.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::XYh1D) = m.site

function bond_term(m::XYh1D, i::Int, j::Int)
    H = OpSum()
    H += -m.J * (1 + m.γ), "Sx", i, "Sx", j
    H += -m.J * (1 - m.γ), "Sy", i, "Sy", j
    H += -(m.h / 2), "Sz", i
    H += -(m.h / 2), "Sz", j
    return H
end

function boundary_patch(m::XYh1D, k::Int)
    H = OpSum()
    H += -(m.h / 2), "Sz", k
    return H
end

function bond_coupling_term(m::XYh1D, i::Int, j::Int)
    H = OpSum()
    H += -m.J * (1 + m.γ), "Sx", i, "Sx", j
    H += -m.J * (1 - m.γ), "Sy", i, "Sy", j
    return H
end

function onsite_term(m::XYh1D, k::Int)
    H = OpSum()
    H += -m.h, "Sz", k
    return H
end

function onsite_observable_op(::XYh1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("XYh1D: unsupported onsite observable $name")
end
