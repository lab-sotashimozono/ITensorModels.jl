module ITensorModels

using ITensors
using ITensorMPS
using ITensorSiteKit: PhysSite

export AbstractLatticeModel, site_type
export bond_term, boundary_patch, local_ham_terms, build_opsum
export bond_coupling_term, onsite_term
export onsite_observable_op, build_onsite_observable_opsum
export TFIM, TFIML, XXZ1D, Heisenberg1D, KitaevBond, LatticeModel
# Site-dependent (disordered) transverse-field Ising chain — a seed fixes the
# couplings, so every representation downstream gets the SAME Hamiltonian.
export RandomTFIM, random_tfim, rescale_disorder
export XYh1D
export LongRangeIsing1D
export ExtendedHubbard1D
export Compass1D
export Hubbard1D
export AndersonImpurity1D, semielliptic_anderson, star_to_chain
export AKLT1D
export TightBindingV1D
export J1J2Heisenberg1D
export BoseHubbard1D
export DMIHeisenberg1D
export LongRangeXY1D
export S1AnisotropicD1D
export PXP1D
export RiceMele1D
export RiceMeleHubbard1D
export SSH1D
export S1XXZ1D
export SpinHalfXYZ1D
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
export bond_displacement
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
    bond_displacement(model) -> Float64

The distance between the two sites a bond of `model` connects, in the length unit `model`
measures its vector potential `A` in. This is the whole of what a Peierls substitution needs
from a model: the phase on a bond is `A * bond_displacement(model)`, which is
`AbstractQAtlas.peierls_phase` contracted with that displacement.

It takes no site indices ON PURPOSE. A model here has one hop distance, and the obvious
generalisation — `j - i` — would be wrong: on the `LatticeCore` path the indices are MPS
POSITIONS produced by a user-supplied ordering, so their difference is not a displacement.

Every chain here places its sites one unit apart, so this is `1.0`: `A` is measured per SITE
spacing. That is a CHOICE, and it is written down because the checks that would normally
catch a wrong one are structurally blind to it:

| check | why it cannot see the unit |
|:--|:--|
| static / equilibrium, `A = 0` | `H` depends on `A` only through `A * d` |
| anything on an OPEN chain | a uniform `A` is a pure gauge there |
| a comparison against an oracle built the same way | it shares the convention |

Measured on an 8-site Rice-Mele chain, `v, w, Δ = 0.7, 1.3, 0.4`: open, `E(A) - E(0)` stays
under `1.1e-14` out to `A = 1`; closed into a ring, `d²E/dA²` at `A = 0` comes out in the
ratio `3.999981` between `d = 1` and `d = 1/2`. So the unit is invisible until a fixture is
periodic, and then an `n`-th order response carries a factor `dⁿ`.

Sources differ, so a comparison has to convert. Ono, *Phys. Rev. Lett.* **135**, 026401
(2025) sets the nearest-neighbour distance to 1/2 — `A` per UNIT CELL, `e^{iA/2}` on a bond —
so reproducing it with these models means passing `A = A_paper / 2`.

A current operator carries the same factor: `J = -∂H/∂A` brings down one power of the
displacement, so it has to be built from this same number rather than a second guess at it.
"""
function bond_displacement end

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
include("models/random_tfim.jl")
include("models/tfiml.jl")
include("models/xxz.jl")
include("models/heisenberg.jl")
include("models/heisenberg_s1.jl")
include("models/hubbard_1d.jl")
include("models/anderson_impurity_1d.jl")
include("models/aklt_1d.jl")
include("models/tightbinding_v_1d.jl")
include("models/j1j2_heisenberg_1d.jl")
include("models/bose_hubbard_1d.jl")
include("models/dmi_heisenberg_1d.jl")
include("models/long_range_xy_1d.jl")
include("models/s1_anisotropic_d.jl")
include("models/pxp_1d.jl")
include("models/rice_mele_1d.jl")
include("models/rice_mele_hubbard_1d.jl")
include("models/ssh_1d.jl")
include("models/s1_xxz_1d.jl")
include("models/spin_half_xyz_1d.jl")
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
