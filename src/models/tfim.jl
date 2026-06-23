using ITensors: SiteType, @SiteType_str

"""
    TFIM(; J = 1.0, h = 1.0, site = SiteType("S=1/2"))

1D transverse-field Ising model

    H = -J Σᵢ Zᵢ Zᵢ₊₁ - h Σᵢ Xᵢ

where `Z`, `X` are the natural local operators of `site`:

- `SiteType("S=1/2")` → `Z = Sᶻ`, `X = Sˣ` (ITensors `"Sz"` / `"Sx"`).
- `SiteType("Qubit")` → `Z = σᶻ`, `X = σˣ` (ITensors `"Z"` / `"X"`).

Callers pick the `SiteType` to match the physical units they want `J`
and `h` to carry. QAtlas conversion happens in `ext/QAtlasExt.jl`.
"""
Base.@kwdef struct TFIM <: AbstractLatticeModel
    J::Float64 = 1.0
    h::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::TFIM) = m.site

# Per-site operator names selected by the SiteType.
# S=1/2, S=1, S=3/2 share the ITensors spin operator names "Sz" / "Sx";
# the actual Hilbert dimension / operator matrix is resolved later when
# the OpSum is materialised into an MPO on specific site indices.
ising_z_op(::SiteType"S=1/2") = "Sz"
ising_x_op(::SiteType"S=1/2") = "Sx"
ising_z_op(::SiteType"S=1") = "Sz"
ising_x_op(::SiteType"S=1") = "Sx"
ising_z_op(::SiteType"S=3/2") = "Sz"
ising_x_op(::SiteType"S=3/2") = "Sx"
ising_z_op(::SiteType"Qubit") = "Z"
ising_x_op(::SiteType"Qubit") = "X"

function onsite_observable_op(m::TFIM, name::Symbol)
    # TFIM's symmetry-relevant onsite observables are Sˣ (order parameter
    # of the disordered/paramagnetic regime) and Sᶻ (of the broken-symmetry
    # regime). Sʸ is not conventionally used for TFIM but available for
    # completeness on S=1/2.
    s = site_type(m)
    name === :sx && return ising_x_op(s)
    name === :sz && return ising_z_op(s)
    if name === :sy
        return _tfim_y_op(s)
    end
    return error("TFIM: unsupported onsite observable $name on site $s")
end

_tfim_y_op(::SiteType"S=1/2") = "Sy"
_tfim_y_op(::SiteType"S=1") = "Sy"
_tfim_y_op(::SiteType"S=3/2") = "Sy"
_tfim_y_op(::SiteType"Qubit") = "Y"

function bond_term(m::TFIM, i::Int, j::Int)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.J, zop, i, zop, j
    opsum += -m.h / 2, xop, i
    opsum += -m.h / 2, xop, j
    return opsum
end

function boundary_patch(m::TFIM, i::Int)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h / 2, xop, i
    return opsum
end

# ---------------------------------------------------------------------
# Split protocol: pure bond coupling and pure on-site, used by
# `ModulatedModel` to apply per-bond / per-site envelopes (SSD, etc.)
# ---------------------------------------------------------------------

function bond_coupling_term(m::TFIM, i::Int, j::Int)
    zop = ising_z_op(m.site)
    opsum = OpSum()
    opsum += -m.J, zop, i, zop, j
    return opsum
end

function onsite_term(m::TFIM, k::Int)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h, xop, k
    return opsum
end
