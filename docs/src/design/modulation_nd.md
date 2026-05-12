# ND Modulation — Formalism & API

Status: agreed 2026-05-12. This document describes the formalism
for the ND modulation extension implemented across the
`feat/modulation-2d-*` PR chain. Each section is labelled with the
PR that introduces the corresponding code.

## Goal

Lift the existing 1D `AbstractModulation` machinery
(`SSD`, `SinPower{N}`, `SmoothBoundary`, `Tabulated`, `ModulatedModel`)
to arbitrary-dimensional lattices (honeycomb, square, triangular, …)
without disturbing the 1D code path.

## Geometry of an envelope

An envelope `f : ℝ^D → [0, 1]` is decomposed as

```
f(r) = g( d(r ; r_c) )
```

where

- `r_c ∈ ℝ^D` is a **center** (see `AbstractCenter`),
- `d : ℝ^D → ℝ_+` (or `ℝ^D → ℝ_+^D` for the axis-product variant) is a
  **distance metric** (see `AbstractDistance`),
- `g : ℝ_+ → [0, 1]` (or `g : ℝ_+^D → [0, 1]` for axis-product) is a
  **profile** (see `AbstractProfile`).

The `AxisProductDistance` path treats each axis independently and
combines per-axis profiles by `f(r) = Π_d g(d_d; R_d)` — this recovers
the LatticeCore-style hypercubic SSD `Π_d sin²(π (c_d − 1/2) / L_d)`.

## Type hierarchy

```
AbstractModulation                                  # existing (1D)
└── AbstractModulationND                            # new

AbstractCenter
├── GeometricCenter                                 # mean of positions
├── BoundingBoxCenter                               # (min + max) / 2 per axis
└── ExplicitCenter(r_c::SVector{D,T})

AbstractDistance
├── EuclideanDistance                               # ‖r − r_c‖₂      (sphere)
├── AxialDistance(axis::Int)                        # |r[a] − r_c[a]| (cylinder axial)
├── PerpendicularDistance(axis::Int)                # ‖r[¬a] − r_c[¬a]‖₂ (cylinder radial)
└── AxisProductDistance                             # NTuple (|r[d] − r_c[d]|)_d

AbstractProfile
├── SinSquareProfile(R)        # = 1 − sin²(π d / (2R))  ≡ cos²(π d / (2R))
├── SinPowerProfile{N}(R)      # = 1 − sin^N(π d / (2R))   ← N=2 ⇒ SinSquare
└── CosineRampProfile(R, edge) # plateau + cosine ramp (Vekic–White on a disk)

RadialEnvelope{C,D,P} <: AbstractModulationND        # composition wrapper
    center::C
    distance::D
    profile::P
```

For `AxisProductDistance` the profile carries an `SVector` of per-axis
radii, e.g. `SinSquareProfile(SVector(Rx, Ry))`.

## Profile formulas

The profile `g` is defined so that **`g(0) = 1`**, **`g(R) = 0`**, and
`g(d) = 0` for `d > R`.

### SinSquare

```math
g(d) =
\begin{cases}
1 - \sin^2(\pi d / (2R)) & 0 \le d \le R \
0 & d > R
\end{cases}
\;\;\equiv\;\; \cos^2(\pi d / (2R)) \;\; \text{on } [0, R]
```

### SinPower{N}

```math
g(d) =
\begin{cases}
1 - \sin^N(\pi d / (2R)) & 0 \le d \le R \
0 & d > R
\end{cases}
```

- `N = 2` reduces to `SinSquare`.
- Larger `N` makes the bulk plateau **wider** and the boundary fall-off
  steeper — the radial counterpart of `sin^N(π i / L)` in 1D.
- **Not** `cos^N`. The naive radial transliteration `g(d) = cos^N(π d/(2R))`
  reverses the semantics (larger `N` makes the bulk shrink); the
  `1 − sin^N` form preserves the 1D meaning.

### CosineRamp(R, edge)

```math
g(d) =
\begin{cases}
1 & d \le R - \mathrm{edge} \
\frac{1 + \cos(\pi (d - (R - \mathrm{edge})) / \mathrm{edge})}{2}
                                & R - \mathrm{edge} < d < R \
0 & d \ge R
\end{cases}
```

Generalises Vekic–White smooth boundary conditions to a disk / hyper-ball.

## Evaluation contract

For a lattice `lat` of `AbstractLattice{D,T}`:

```
center_position(c::AbstractCenter, lat) -> SVector{D,T}
distance_at(d::AbstractDistance, lat, k::Int, r_c::SVector) -> Real (or NTuple{D,Real})
profile_value(p::AbstractProfile, d::Real) -> Float64
profile_value(p::AbstractProfile, ds::NTuple{D,Real}) -> Float64   # axis-product only
site_envelope(env::RadialEnvelope, lat, k::Int) -> Float64
```

### Site weight

```
site_weight(env, lat, k) = site_envelope(env, lat, k)
                        = profile_value(env.profile, distance_at(env.distance, lat, k, r_c))
```

### Bond weight (★ midpoint evaluation)

```
bond_weight(env, lat, i, j) = profile_value(
    env.profile,
    distance_at_position(env.distance, (position(lat,i) + position(lat,j)) / 2, r_c),
)
```

The midpoint convention matches the existing 1D rule
`bond_weight(SSD, i, L) = sin²(π i / L)` (site is at half-integer `i − 1/2`,
bond midpoint at integer `i`). This **differs** from LatticeCore's
`bond_weight(::SSD, lat, i, j) = (f(r_i) + f(r_j))/2`; the two agree to
leading order for smooth envelopes but disagree on the strict numerics.

## Hamiltonian assembly (`ModulatedLatticeModel`)

```
local_ham_terms(m::ModulatedLatticeModel{LatticeModel,Env}, _; boundary=:full):
    terms = OpSum[]
    for b in bonds(m.base.lattice):
        push!(terms, bond_weight(env, lat, b.i, b.j)
                     * bond_coupling_term(submodel(b), ord[b.i], ord[b.j]))
    for k in 1:num_sites(m.base.lattice):
        push!(terms, site_weight(env, lat, k)
                     * onsite_term(submodel_for_site(k), ord[k]))
    return terms
```

ND Hamiltonian assembly **fully separates** bond coupling from on-site
terms — there is no half-distribution and no `boundary_patch` step. This
works for any coordination number and is the principled generalisation
of the 1D protocol.

The 1D `ModulatedModel` is **not modified**; it keeps the
half-distribution + boundary_patch convention. A regression test
checks that `ModulatedLatticeModel(line_lattice(L), radial_ssd_1d)`
produces an `OpSum` equal (up to operator-algebra normalisation) to
`ModulatedModel(TFIM(), L, SSD())`.

## Factories (user-facing)

```julia
rectangular_ssd(lat; N=2)                          # axis-product (LatticeCore SSD); SinSquare at N=2
cylindrical_ssd(lat; axis=1, N=2)                  # axial; uniform on the cylinder
spherical_ssd(lat; radius=:inscribed, N=2)         # EuclideanDistance + Sin{Square,Power{N}}
```

`N` switches between `SinSquareProfile` (the default at `N = 2`) and
`SinPowerProfile{N}` (`N ≥ 3`). Smooth-boundary (Vekic–White) variants
are accessible by constructing `RadialEnvelope(..., CosineRampProfile(R, edge))`
directly; a dedicated `spherical_smooth` factory is not yet shipped.

The `radius` keyword chooses between `:inscribed` (`min(L_d)/2`) and
`:circumscribed` (`‖(L_d / 2)‖₂`).

## File plan

- `src/core/modulation_nd.jl` — all primitives + `RadialEnvelope`
- `src/core/factories_nd.jl` — generic function declarations for the factories
- `src/models/modulated_lattice.jl` — `ModulatedLatticeModel` wrapper
- `ext/LatticeCoreExt.jl` — lattice-bound methods (`center_position`,
  `distance_at`, `site_envelope`, `site_weight`, `bond_weight`,
  `local_ham_terms` for `ModulatedLatticeModel`, plus factory bodies)
- `src/ITensorModels.jl` — includes + exports
- `test/base/test_modulation_nd_core.jl` — primitives unit tests (lattice-free)
- `test/base/test_modulation_nd_lattice.jl` — lattice-bound primitive tests
- `test/base/test_modulated_lattice.jl` — Hamiltonian build on honeycomb
- `test/base/test_modulation_nd_1d_equivalence.jl` — ND-on-LineLattice ≡ 1D regression
- `test/base/test_modulation_nd_factories.jl` — user-facing factory tests

## Decision log (agreed 2026-05-12)

- (D1) `SinPower` formula: `1 − sin^N(π d / (2R))` (preserves 1D semantics).
- (D2) `bond_weight` evaluated at the **bond midpoint** position.
- (D3) ND `ModulatedLatticeModel` **fully separates** bond / onsite emission.
- (D4) `AxisProductDistance` carries per-axis radii via `SVector`.
- (D5) `AxisProductDistance` is a distinct path; not composable with
   `EuclideanDistance` / `AxialDistance` in the same envelope.
- (D6) Profile sweeping over multiple distance types: deferred.
