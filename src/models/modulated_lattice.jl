"""
    ModulatedLatticeModel(base::AbstractLatticeModel, envelope::AbstractModulationND)
    modulated_lattice(base; envelope)

Apply an ND spatial envelope (an [`AbstractModulationND`](@ref) such as
[`RadialEnvelope`](@ref)) to a [`LatticeModel`](@ref). The Hamiltonian
on the lattice is

```
H = Σ_{(i,j) ∈ bonds(lattice)}
        bond_weight(env, lattice, i, j) · bond_coupling_term(sub(b), ord[i], ord[j])
  + Σ_{k ∈ sites(lattice)}
        site_weight(env, lattice, k) · onsite_term(sub_site(k), ord[k])
```

ND Hamiltonian assembly **fully separates** bond coupling from on-site
terms; there is no half-distribution and no `boundary_patch`. This
generalises the 1D protocol to arbitrary coordination numbers
(honeycomb / kagome / dice / shastry-sutherland / …) without further
case analysis.

The bond submodels are looked up from `base.bond_models` exactly as the
plain `LatticeModel` does; the same submodel supplies `bond_coupling_term`
for the bond it owns. Onsite terms are emitted once per site using the
submodel attached to any bond touching that site (current
implementation requires site-uniform onsite terms — which is the case
for all bundled `TFIM` / `TFIML` / `XXZ1D` / `Heisenberg1D` models).

Per-site model overrides are not yet supported; if a future model
carries site-dependent fields, add an `onsite_submodel_for_site`
accessor and dispatch on it.

# Examples

```julia
using ITensorModels, LatticeCore, Lattice2D

lat  = honeycomb(6, 6; boundary = OpenAxis())
base = LatticeModel(; lattice = lat, bond_models = Dict(:nearest => Heisenberg1D(J = 1.0)))
env  = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(7.0))
mod  = modulated_lattice(base; envelope = env)
# H = build_opsum(mod, sites)
```
"""
struct ModulatedLatticeModel{LM<:AbstractLatticeModel,Env<:AbstractModulationND} <:
       AbstractLatticeModel
    base::LM
    envelope::Env
end

"""
    modulated_lattice(base::AbstractLatticeModel; envelope::AbstractModulationND)

Convenience constructor for [`ModulatedLatticeModel`](@ref).
"""
function modulated_lattice(
    base::AbstractLatticeModel; envelope::AbstractModulationND
)
    return ModulatedLatticeModel(base, envelope)
end

# Site type / observable lookup transparently delegate to the base
# `LatticeModel` — the envelope only rescales the energetics, not the
# physical Hilbert space or which operator measures each observable.

site_type(m::ModulatedLatticeModel) = site_type(m.base)

function onsite_observable_op(m::ModulatedLatticeModel, name::Symbol)
    return onsite_observable_op(m.base, name)
end
