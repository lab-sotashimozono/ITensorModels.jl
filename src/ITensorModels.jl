module ITensorModels

using ITensors
using ITensorMPS
using ITensorSiteKit: PhysSite

export AbstractLatticeModel, site_type
export bond_term, boundary_patch, local_ham_terms, build_opsum
export bond_coupling_term, onsite_term
export onsite_observable_op, build_onsite_observable_opsum
export TFIM, TFIML, XXZ1D, Heisenberg1D, KitaevBond, LatticeModel
export XYh1D
export LongRangeIsing1D
export ExtendedHubbard1D
export Compass1D
export Hubbard1D
export AKLT1D
export TightBindingV1D
export J1J2Heisenberg1D
export BoseHubbard1D
export DMIHeisenberg1D
export LongRangeXY1D
export PXP1D
export Cluster1D
export TFIM, TFIML, XXZ1D, Heisenberg1D, S1Heisenberg1D, KitaevBond, LatticeModel
export TFIM, TFIML, XXZ1D, Heisenberg1D, KitaevBond, TightBinding1D, LatticeModel
export AbstractModulation, Uniform, SSD, SinPower, SmoothBoundary, Tabulated
export site_weight, bond_weight
export ModulatedModel, modulated
export ModulatedLatticeModel, modulated_lattice
export AbstractModulationND
export AbstractCenter, GeometricCenter, BoundingBoxCenter, ExplicitCenter
export AbstractDistance,
    EuclideanDistance, AxialDistance, PerpendicularDistance, AxisProductDistance
export AbstractProfile, SinSquareProfile, SinPowerProfile, CosineRampProfile
export RadialEnvelope
export distance_at_position, distance_at, center_position, profile_value, site_envelope
export spherical_ssd, cylindrical_ssd, rectangular_ssd
export to_qatlas, from_qatlas

"""
    AbstractLatticeModel

Root type for Hamiltonian specifications. Concrete subtypes implement
[`bond_term`](@ref) (and optionally [`boundary_patch`](@ref)); the
generic [`local_ham_terms`](@ref) / [`build_opsum`](@ref) machinery
lifts those into a full MPO OpSum for any site layout.
"""
abstract type AbstractLatticeModel end

"""
    site_type(model) -> ITensors.SiteType

ITensors `SiteType` used when building physical indices for `model`.
"""
function site_type end

"""
    to_qatlas(model)

Translate an `ITensorModels` model to the matching `QAtlas` model,
applying the unit conversion implied by `model.site`. Implemented in
`ext/QAtlasExt.jl` when `QAtlas` is loaded.
"""
function to_qatlas end

"""
    from_qatlas(qmodel)

Inverse of [`to_qatlas`](@ref). Implemented in `ext/QAtlasExt.jl`.
"""
function from_qatlas end

include("core/interface.jl")
include("core/observables.jl")
include("core/modulation.jl")
include("core/modulation_nd.jl")
include("core/factories_nd.jl")

include("models/tfim.jl")
include("models/tfiml.jl")
include("models/xxz.jl")
include("models/heisenberg.jl")
include("models/heisenberg_s1.jl")
include("models/hubbard_1d.jl")
include("models/aklt_1d.jl")
include("models/tightbinding_v_1d.jl")
include("models/j1j2_heisenberg_1d.jl")
include("models/bose_hubbard_1d.jl")
include("models/dmi_heisenberg_1d.jl")
include("models/long_range_xy_1d.jl")
include("models/pxp_1d.jl")
include("models/cluster_1d.jl")
include("models/kitaev_bond.jl")
include("models/xy_h_1d.jl")
include("models/long_range_ising_1d.jl")
include("models/extended_hubbard_1d.jl")
include("models/compass_1d.jl")
include("models/tightbinding_1d.jl")
include("models/lattice_model.jl")
include("models/modulated.jl")
include("models/modulated_lattice.jl")

end # module ITensorModels
