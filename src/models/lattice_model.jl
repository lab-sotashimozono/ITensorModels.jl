"""
    LatticeModel(; lattice, bond_models, ordering=Int[])

Lattice-graph-based Hamiltonian composed from existing 1D / bond-local
model primitives. Works on any `LatticeCore.AbstractLattice` — so
`Lattice2D.Lattice` (Square / Honeycomb / Kagome / …) and
`QuasiCrystal.QuasicrystalData` (Fibonacci / Penrose / Ammann-Beenker)
plug in uniformly.

Fields

- `lattice::L` — the `LatticeCore.AbstractLattice` whose sites and bonds
  describe the connectivity. Both endpoints of every bond become
  coupled in the emitted MPO.
- `bond_models::D` — mapping from `Bond.type` (a `Symbol` carried on
  every `LatticeCore.Bond`) to an `AbstractLatticeModel` whose
  `bond_term(submodel, i, j)` supplies the per-bond OpSum. Anisotropic
  lattices such as Shastry-Sutherland can give different couplings
  per bond type via this dict.
- `ordering::Vector{Int}` — 1D embedding. `ordering[i]` is the MPS
  position of lattice site `i`. An empty default selects the natural
  order `1:num_sites(lattice)`. Pass a snake / Hilbert-curve order here
  when you want nearby MPS positions to correspond to nearby lattice
  positions.

Site type is inherited from the bond models — they must all agree on
`site_type`.

The actual `build_opsum` / `local_ham_terms` methods require
`LatticeCore` to be loaded and live in `ext/LatticeCoreExt.jl`.
"""
struct LatticeModel{L,D} <: AbstractLatticeModel
    lattice::L
    bond_models::D
    ordering::Vector{Int}

    function LatticeModel(
        lattice::L, bond_models::D; ordering::Vector{Int}=Int[]
    ) where {L,D}
        return new{L,D}(lattice, bond_models, ordering)
    end
end

function LatticeModel(; lattice, bond_models, ordering::Vector{Int}=Int[])
    return LatticeModel(lattice, bond_models; ordering)
end

function site_type(m::LatticeModel)
    sts = unique(site_type(bm) for bm in values(m.bond_models))
    length(sts) == 1 ||
        error("LatticeModel: bond_models carry inconsistent site_type ($sts).")
    return first(sts)
end
