using ITensors: SiteType, @SiteType_str

"""
    TFIML(; J=1.0, h_x=1.0, h_z=0.01, site=SiteType("S=1/2"))

1D transverse-field Ising model with an additional **longitudinal**
field

    H = -J Σ Zᵢ Zᵢ₊₁  -  h_x Σ Xᵢ  -  h_z Σ Zᵢ

`h_x` is the familiar transverse field; `h_z` tilts the `Z` axis
(breaking integrability and pinning the symmetry-broken sector at
`h_x < J`). Operators follow `site` the same way as [`TFIM`](@ref).
"""
Base.@kwdef struct TFIML <: AbstractLatticeModel
    J::Float64 = 1.0
    h_x::Float64 = 1.0
    h_z::Float64 = 0.01
    site::SiteType = SiteType("S=1/2")
end

site_type(m::TFIML) = m.site

function onsite_observable_op(m::TFIML, name::Symbol)
    return onsite_observable_op(TFIM(; site=m.site), name)
end

function bond_term(m::TFIML, i::Int, j::Int)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.J, zop, i, zop, j
    opsum += -m.h_x / 2, xop, i
    opsum += -m.h_x / 2, xop, j
    opsum += -m.h_z / 2, zop, i
    opsum += -m.h_z / 2, zop, j
    return opsum
end

function boundary_patch(m::TFIML, i::Int)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h_x / 2, xop, i
    opsum += -m.h_z / 2, zop, i
    return opsum
end

# ---------------------------------------------------------------------
# Split protocol: bond is the pure ZZ coupling, on-site combines
# transverse and longitudinal fields.
# ---------------------------------------------------------------------

function bond_coupling_term(m::TFIML, i::Int, j::Int)
    zop = ising_z_op(m.site)
    opsum = OpSum()
    opsum += -m.J, zop, i, zop, j
    return opsum
end

function onsite_term(m::TFIML, k::Int)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h_x, xop, k
    opsum += -m.h_z, zop, k
    return opsum
end
