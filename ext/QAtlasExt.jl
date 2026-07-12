module QAtlasExt

using ITensorModels
using ITensors: SiteType, @SiteType_str
using QAtlas

# --- Model bridge -------------------------------------------------------

"""
    to_qatlas(m::ITensorModels.TFIM) -> QAtlas.TFIM

Translate to the QAtlas (Pauli-convention) struct, applying the local
operator convention coming from `m.site`. Register a
`to_qatlas(m, ::SiteType"...")` method to extend support.
"""
ITensorModels.to_qatlas(m::ITensorModels.TFIM) = ITensorModels.to_qatlas(m, m.site)

# Spin-½ site: our H = -J Σ SzSz - h Σ Sx = -(J/4) Σ σzσz - (h/2) Σ σx.
function ITensorModels.to_qatlas(m::ITensorModels.TFIM, ::SiteType"S=1/2")
    return QAtlas.TFIM(; J=(m.J / 4), h=(m.h / 2))
end

# Qubit site: operators already Pauli, no rescaling.
function ITensorModels.to_qatlas(m::ITensorModels.TFIM, ::SiteType"Qubit")
    return QAtlas.TFIM(; J=m.J, h=m.h)
end

"""
    from_qatlas(qm::QAtlas.TFIM) -> ITensorModels.TFIM

Inverse of [`to_qatlas`](@ref). Defaults to a `Qubit`-site struct so
`J`/`h` carry the same Pauli units as `qm`.
"""
function ITensorModels.from_qatlas(qm::QAtlas.TFIM)
    return ITensorModels.TFIM(; J=qm.J, h=qm.h, site=SiteType("Qubit"))
end

# --- XXZ1D --------------------------------------------------------------

ITensorModels.to_qatlas(m::ITensorModels.XXZ1D) = ITensorModels.to_qatlas(m, m.site)

# QAtlas XXZ1D uses spin-½ convention, so the coefficients pass through.
function ITensorModels.to_qatlas(m::ITensorModels.XXZ1D, ::SiteType"S=1/2")
    return QAtlas.XXZ1D(m.J, m.Δ)
end

function ITensorModels.from_qatlas(qm::QAtlas.XXZ1D)
    return ITensorModels.XXZ1D(; J=qm.J, Δ=qm.Δ)
end

# --- Heisenberg1D -------------------------------------------------------

# QAtlas.Heisenberg1D is parameter-free (J = 1 fixed). Map either way
# only when the ITensorModels side is on a S=1/2 site with J = 1.
ITensorModels.to_qatlas(m::ITensorModels.Heisenberg1D) = ITensorModels.to_qatlas(m, m.site)

function ITensorModels.to_qatlas(m::ITensorModels.Heisenberg1D, ::SiteType"S=1/2")
    return QAtlas.Heisenberg1D()
end

function ITensorModels.from_qatlas(::QAtlas.Heisenberg1D)
    return ITensorModels.Heisenberg1D()
end

# --- S1Heisenberg1D ----------------------------------------------------

function ITensorModels.to_qatlas(m::ITensorModels.S1Heisenberg1D)
    return ITensorModels.to_qatlas(m, m.site)
end

# QAtlas.S1Heisenberg1D uses the same J coefficient on spin-1 Sx/Sy/Sz
# matrices (eigenvalues -1, 0, +1) -- no rescaling needed.
function ITensorModels.to_qatlas(m::ITensorModels.S1Heisenberg1D, ::SiteType"S=1")
    return QAtlas.S1Heisenberg1D(; J=m.J)
end

function ITensorModels.from_qatlas(qm::QAtlas.S1Heisenberg1D)
    return ITensorModels.S1Heisenberg1D(; J=qm.J)
end

# --- TFIML → E8 (critical Ising + longitudinal field) -------------------
#
# The transverse-field Ising model at its CRITICAL transverse field,
# perturbed by a small longitudinal field `h_z`, flows to the Zamolodchikov
# E8 integrable field theory: 8 particles with masses fixed by E8, with
# `m₂/m₁ = 2 cos(π/5) = φ` (golden ratio; observed in CoNb₂O₆). `QAtlas.E8`
# is parameter-free and exposes the universal mass-RATIO spectrum via
# `fetch(E8(), E8Spectrum(), Infinite())`. Those ratios are the `h_z → 0`
# universal limit; a finite-`h_z` lattice model only approaches them.

ITensorModels.to_qatlas(m::ITensorModels.TFIML) = ITensorModels.to_qatlas(m, m.site)

# Qubit (Pauli): H = -J Σσᶻσᶻ - h_x Σσˣ - h_z Σσᶻ; Ising-critical at h_x = J.
function ITensorModels.to_qatlas(m::ITensorModels.TFIML, ::SiteType"Qubit")
    _warn_if_not_e8(m.h_x, m.J, m.h_z)
    return QAtlas.E8()
end

# S=1/2 (Sᶻ,Sˣ = σ/2): Pauli couplings J/4, h_x/2 ⇒ critical at h_x = J/2.
function ITensorModels.to_qatlas(m::ITensorModels.TFIML, ::SiteType"S=1/2")
    _warn_if_not_e8(m.h_x, m.J / 2, m.h_z)
    return QAtlas.E8()
end

# E8 universality holds only at the critical transverse field, and the mass
# ratios are the h_z → 0 limit. Warn (do not hard-fail) so a caller can still
# fetch the reference ratios while being told when they are off-regime.
function _warn_if_not_e8(h_x, h_x_crit, h_z)
    isapprox(h_x, h_x_crit; rtol=0.02) || @warn(
        "TFIML→E8: the E8 spectrum requires the critical transverse field " *
            "(h_x ≈ $(h_x_crit) for this site); got h_x=$(h_x). Returned E8 ratios may not apply.",
        maxlog = 1,
    )
    (0 < h_z ≤ 0.2) || @warn(
        "TFIML→E8: the E8 mass ratios are the h_z→0 universal limit; got h_z=$(h_z). " *
            "A finite/large h_z only approximates them.",
        maxlog = 1,
    )
    return nothing
end

# Inverse: a canonical critical TFIML with a small longitudinal perturbation.
function ITensorModels.from_qatlas(::QAtlas.E8; J=1.0, h_z=0.01, site=SiteType("Qubit"))
    h_x = site === SiteType("S=1/2") ? J / 2 : J
    return ITensorModels.TFIML(; J=J, h_x=h_x, h_z=h_z, site=site)
end

# --- fetch forwarder ----------------------------------------------------
#
# Route `QAtlas.fetch` calls that take an ITensorModels model through
# `to_qatlas`.  Every concrete-struct method
# `fetch(::QAtlas.Model, ::Quantity, ::BC; ...)` registered inside QAtlas
# becomes available for free on any `AbstractLatticeModel` whose site
# has a `to_qatlas` implementation.  Adding a method to `QAtlas.fetch`
# with an ITensorModels-owned argument type is not type piracy.

function QAtlas.fetch(
    m::ITensorModels.AbstractLatticeModel,
    q::QAtlas.AbstractQuantity,
    bc::QAtlas.BoundaryCondition;
    kwargs...,
)
    return QAtlas.fetch(ITensorModels.to_qatlas(m), q, bc; kwargs...)
end

end # module QAtlasExt
