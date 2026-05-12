# ND modulation envelopes — primitives for honeycomb / square / kagome / ...
# See `docs/src/design/modulation_nd.md` for the formalism.

"""
    AbstractModulationND <: AbstractModulation

ND extension of [`AbstractModulation`](@ref). ND modulations expose

    site_weight(mod, lat, k::Int)         ::Float64
    bond_weight(mod, lat, i::Int, j::Int) ::Float64

dispatching on the lattice and a single site/bond index, rather than on
the `(i, L)` chain coordinates used by the 1D protocol. The two
signature families share `AbstractModulation` as their supertype but
are otherwise independent — existing 1D code is unaffected.

Lattice-bound evaluation methods (`center_position`, `distance_at`,
`site_weight(::AbstractModulationND, lat, k)`, …) live in the
`LatticeCoreExt` extension so that the core module stays free of a
`LatticeCore` dependency.
"""
abstract type AbstractModulationND <: AbstractModulation end

# ---------------------------------------------------------------------
# Center
# ---------------------------------------------------------------------

"""
    AbstractCenter

Strategy for choosing the envelope's center `r_c ∈ ℝ^D`. Concrete
subtypes implement `center_position(c, lat) -> SVector{D, T}` in the
`LatticeCoreExt` extension.
"""
abstract type AbstractCenter end

"""
    GeometricCenter()

Center at the arithmetic mean of all site positions,
`r_c = (1/N) Σ_k r_k`.
"""
struct GeometricCenter <: AbstractCenter end

"""
    BoundingBoxCenter()

Center at the midpoint of the lattice's axis-aligned bounding box,
`r_c = (min(r_k) + max(r_k)) / 2` per axis.
"""
struct BoundingBoxCenter <: AbstractCenter end

"""
    ExplicitCenter(r_c)

Use the user-supplied vector `r_c` as the envelope center. Accepts any
indexable container (`SVector`, `Vector`, tuple).
"""
struct ExplicitCenter{V} <: AbstractCenter
    r::V
end

# ---------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------

"""
    AbstractDistance

Distance metric carried by a [`RadialEnvelope`](@ref). Concrete subtypes
implement `distance_at_position(d, r, r_c)` (lattice-free) **or**
`distance_at(d, lat, k, r_c)` (lattice-bound, in `LatticeCoreExt`).
The return value is either a `Real` or an `NTuple{D, Real}` (the latter
only for [`AxisProductDistance`](@ref)).
"""
abstract type AbstractDistance end

"""
    EuclideanDistance()

`d(r) = ‖r − r_c‖₂`. Use for spherical / hyper-spherical envelopes.
"""
struct EuclideanDistance <: AbstractDistance end

"""
    AxialDistance(axis::Int)

`d(r) = |r[axis] − r_c[axis]|`. Single-axis distance — produces a slab-
shaped envelope that is uniform in every other direction, the natural
generalisation of 1D SSD along one axis of a cylinder.
"""
struct AxialDistance <: AbstractDistance
    axis::Int

    function AxialDistance(axis::Int)
        axis >= 1 || error("AxialDistance: axis must be >= 1, got $axis")
        return new(axis)
    end
end

"""
    PerpendicularDistance(axis::Int)

`d(r) = ‖r[¬axis] − r_c[¬axis]‖₂`. Distance restricted to the
hyperplane orthogonal to `axis`. Produces a tube-shaped envelope along
`axis` — the canonical cylindrical SSD when `axis` is the cylinder
axis.
"""
struct PerpendicularDistance <: AbstractDistance
    axis::Int

    function PerpendicularDistance(axis::Int)
        axis >= 1 || error("PerpendicularDistance: axis must be >= 1, got $axis")
        return new(axis)
    end
end

"""
    AxisProductDistance()

Per-axis tuple `d = (|r[1] − r_c[1]|, …, |r[D] − r_c[D]|)`. Combined
with a profile carrying per-axis radii this recovers the LatticeCore
hypercubic SSD `Π_d sin²(π (c_d − 1/2) / L_d)`.

`AxisProductDistance` is a distinct evaluation path: the matching
[`profile_value`](@ref) overload takes a tuple of per-axis distances
and reduces by product. It cannot be composed with the scalar profile
methods used by [`EuclideanDistance`](@ref) /
[`AxialDistance`](@ref) / [`PerpendicularDistance`](@ref).
"""
struct AxisProductDistance <: AbstractDistance end

"""
    distance_at_position(dist::AbstractDistance, r, r_c) -> Real or NTuple

Evaluate the distance metric at a generic position `r` (not necessarily
a lattice site). Used both for site-resolved evaluation (`r = r_k`) and
midpoint-evaluated bond weights (`r = (r_i + r_j) / 2`).

The lattice-bound variant `distance_at(dist, lat, k, r_c)` is the one
implementations of `site_weight` / `bond_weight` typically call; it
defers to `distance_at_position(dist, position(lat, k), r_c)` and lives
in `LatticeCoreExt` together with the `LatticeCore` import.
"""
function distance_at_position end

# Pure-vector implementations (no lattice). Generic over indexable
# coordinate containers (`SVector`, `Vector`, `Tuple`, …) so they work
# from both the core module and the extension.

function distance_at_position(::EuclideanDistance, r, r_c)
    s = float(zero(promote_type(eltype(r), eltype(r_c))))
    @inbounds for d in eachindex(r)
        δ = r[d] - r_c[d]
        s += δ * δ
    end
    return sqrt(s)
end

function distance_at_position(d::AxialDistance, r, r_c)
    return abs(r[d.axis] - r_c[d.axis])
end

function distance_at_position(d::PerpendicularDistance, r, r_c)
    a = d.axis
    s = float(zero(promote_type(eltype(r), eltype(r_c))))
    @inbounds for k in eachindex(r)
        k == a && continue
        δ = r[k] - r_c[k]
        s += δ * δ
    end
    return sqrt(s)
end

function distance_at_position(::AxisProductDistance, r, r_c)
    return ntuple(d -> abs(r[d] - r_c[d]), length(r))
end

# ---------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------

"""
    AbstractProfile

1D shape function `g : ℝ₊ → [0, 1]` (or `ℝ₊^D → [0, 1]` for the
axis-product path) controlling how the envelope decays away from the
center. Concrete subtypes satisfy `g(0) = 1`, `g(R) = 0`, and `g(d) = 0`
for `d > R`.
"""
abstract type AbstractProfile end

"""
    _validate_radius_positive(name, R)

Reject non-positive profile radii. Scalar `R` must be `> 0`; an
indexable container (e.g. per-axis `SVector` for the axis-product path)
must have every entry `> 0`. Anything else is rejected outright so that
a caller passing e.g. `Complex` or `String` is told immediately rather
than discovering a downstream `MethodError`.
"""
function _validate_radius_positive(name::String, R)
    if R isa Real
        R > 0 || error("$name: R must be > 0, got R=$R")
    elseif applicable(iterate, R)
        all(x -> x isa Real && x > 0, R) ||
            error("$name: every per-axis R entry must be a positive Real, got R=$R")
    else
        error("$name: R must be a positive Real or an iterable of positive Reals, got typeof(R)=$(typeof(R))")
    end
    return nothing
end

"""
    SinSquareProfile(R)

`g(d) = 1 − sin²(π d / (2R)) ≡ cos²(π d / (2R))` on `[0, R]`, zero
outside. The canonical SSD profile.
"""
struct SinSquareProfile{R} <: AbstractProfile
    R::R

    function SinSquareProfile{R}(r::R) where {R}
        _validate_radius_positive("SinSquareProfile", r)
        return new{R}(r)
    end
end

SinSquareProfile(r) = SinSquareProfile{typeof(r)}(r)

"""
    SinPowerProfile{N}(R)

`g(d) = 1 − sin^N(π d / (2R))` on `[0, R]`, zero outside. Reduces to
[`SinSquareProfile`](@ref) at `N = 2`. Larger `N` widens the bulk
plateau and sharpens the boundary fall-off — the radial counterpart of
the 1D `sin^N(π i / L)` envelope used in Hotta–Shibata-style sin-power
modulations.

Defined as `1 − sin^N` rather than the naive `cos^N`: the naive form
flips the semantics of `N` (larger `N` would shrink the bulk plateau).
See `docs/src/design/modulation_nd.md` for the worked-out comparison.
"""
struct SinPowerProfile{N,R} <: AbstractProfile
    R::R

    function SinPowerProfile{N,R}(r::R) where {N,R}
        N isa Integer && N >= 1 ||
            error("SinPowerProfile: type parameter N must be an integer >= 1, got N=$N")
        _validate_radius_positive("SinPowerProfile", r)
        return new{N,R}(r)
    end
end

SinPowerProfile{N}(r::T) where {N,T} = SinPowerProfile{N,T}(r)

"""
    CosineRampProfile(R, edge)

`g(d)` equals `1` on `[0, R − edge]`, falls through a half-cosine ramp
to `0` on `[R − edge, R]`, and is zero on `[R, ∞)`. Generalises
Vekic–White smooth boundary conditions to a disk / hyper-ball.

Currently only supported with scalar `AbstractDistance` (Euclidean /
Axial / Perpendicular). Axis-product variant is deferred.
"""
struct CosineRampProfile{R,E} <: AbstractProfile
    R::R
    edge::E

    function CosineRampProfile{R,E}(r::R, e::E) where {R,E}
        _validate_radius_positive("CosineRampProfile", r)
        e isa Real ||
            error("CosineRampProfile: edge must be Real, got typeof(edge)=$(typeof(e))")
        e > 0 || error("CosineRampProfile: edge must be > 0, got edge=$e")
        # We only check edge <= R for scalar radii; per-axis CosineRamp is
        # deferred so a vector R does not reach this constructor in practice.
        if r isa Real
            e <= r || error(
                "CosineRampProfile: edge must satisfy 0 < edge <= R, got edge=$e, R=$r",
            )
        end
        return new{R,E}(r, e)
    end
end

CosineRampProfile(r, e) = CosineRampProfile{typeof(r),typeof(e)}(r, e)

# ---------------------------------------------------------------------
# profile_value
# ---------------------------------------------------------------------

"""
    profile_value(p::AbstractProfile, d::Real)         -> Float64
    profile_value(p::AbstractProfile, ds::Tuple{...}) -> Float64

Apply the profile to a scalar distance (Euclidean / Axial /
Perpendicular path) or a tuple of per-axis distances (`AxisProduct`
path). The tuple methods expect `p.R` to be an indexable container of
per-axis radii.
"""
function profile_value end

function profile_value(p::SinSquareProfile, d::Real)
    R = float(p.R)
    d >= R && return 0.0
    return 1 - sin(pi * d / (2R))^2
end

function profile_value(p::SinPowerProfile{N}, d::Real) where {N}
    R = float(p.R)
    d >= R && return 0.0
    return 1 - sin(pi * d / (2R))^N
end

function profile_value(p::CosineRampProfile, d::Real)
    R = float(p.R)
    e = float(p.edge)
    d >= R && return 0.0
    d <= R - e && return 1.0
    return (1 + cos(pi * (d - (R - e)) / e)) / 2
end

# AxisProduct path: distance is a tuple, profile carries per-axis radii.
# Bodies inline the same scalar formula used by the `Real`-distance
# methods above; an explicit private helper is not used because the
# expression is short and the duplication is the single point of
# correspondence between the two paths.

function profile_value(p::SinSquareProfile, ds::Tuple{Vararg{Real}})
    val = 1.0
    @inbounds for d in eachindex(ds)
        x, R = float(ds[d]), float(p.R[d])
        val *= x >= R ? 0.0 : 1 - sin(pi * x / (2R))^2
    end
    return val
end

function profile_value(p::SinPowerProfile{N}, ds::Tuple{Vararg{Real}}) where {N}
    val = 1.0
    @inbounds for d in eachindex(ds)
        x, R = float(ds[d]), float(p.R[d])
        val *= x >= R ? 0.0 : 1 - sin(pi * x / (2R))^N
    end
    return val
end

function profile_value(::CosineRampProfile, ::Tuple{Vararg{Real}})
    error(
        "CosineRampProfile combined with AxisProductDistance is not " *
        "supported yet. Use SinSquareProfile or SinPowerProfile for " *
        "axis-product envelopes, or pick a scalar distance metric " *
        "(EuclideanDistance / AxialDistance / PerpendicularDistance) " *
        "for the cosine ramp.",
    )
end

# ---------------------------------------------------------------------
# RadialEnvelope
# ---------------------------------------------------------------------

"""
    RadialEnvelope(center, distance, profile)

Composition of (center × distance × profile) producing an
`AbstractModulationND` envelope:

```
f(r) = profile_value(profile, distance_at_position(distance, r, center_position(center, lat)))
```

Site and bond weights are derived as

```
site_weight(env, lat, k)    = f(position(lat, k))
bond_weight(env, lat, i, j) = f((position(lat, i) + position(lat, j)) / 2)   # midpoint
```

The midpoint convention is a geometric choice on real-space positions
and is the analogue — not a literal port — of the 1D
`bond_weight(SSD, i, L) = sin²(π i / L)` rule, where on the unit-spacing
1D chain the integer bond index `i` happens to coincide with the
Cartesian midpoint of the bond. The two conventions agree on the
unit-spacing 1D chain. The choice of midpoint evaluation (rather than
LatticeCore's two-endpoint arithmetic mean `(f(r_i) + f(r_j)) / 2`) is
recorded as decision D2 in `docs/src/design/modulation_nd.md`; the
prose of that document gives the full motivation.
"""
struct RadialEnvelope{C<:AbstractCenter,D<:AbstractDistance,P<:AbstractProfile} <:
       AbstractModulationND
    center::C
    distance::D
    profile::P

    function RadialEnvelope{C,D,P}(
        c::C, d::D, p::P
    ) where {C<:AbstractCenter,D<:AbstractDistance,P<:AbstractProfile}
        _validate_envelope_compat(d, p)
        return new{C,D,P}(c, d, p)
    end
end

function RadialEnvelope(
    c::C, d::D, p::P
) where {C<:AbstractCenter,D<:AbstractDistance,P<:AbstractProfile}
    return RadialEnvelope{C,D,P}(c, d, p)
end

# Reject AxisProductDistance combined with a scalar-radius profile -- the
# evaluation path indexes p.R[d], which would crash with a raw BoundsError
# on a Float64. Also reject CosineRampProfile + AxisProductDistance since
# the tuple-path profile_value(::CosineRampProfile, ::Tuple) is not yet
# implemented.
function _validate_envelope_compat(d::AxisProductDistance, p::AbstractProfile)
    p isa CosineRampProfile && error(
        "RadialEnvelope: CosineRampProfile combined with AxisProductDistance is " *
        "not supported yet. Use SinSquareProfile / SinPowerProfile with a per-axis " *
        "radius container, or pick a scalar distance (Euclidean / Axial / " *
        "Perpendicular) for the cosine ramp.",
    )
    p.R isa Real && error(
        "RadialEnvelope: AxisProductDistance requires a per-axis radius container " *
        "in the profile, but profile carries a scalar R=$(p.R)::$(typeof(p.R)). " *
        "Pass SinSquareProfile([R1, R2, ...]) (or an SVector), or use the " *
        "rectangular_ssd(lat) factory.",
    )
    return nothing
end

# For scalar distance metrics (Euclidean / Axial / Perpendicular) the
# profile must carry a scalar R; a vector R would index correctly in the
# tuple path but never reach it, producing silently-wrong scalar results.
function _validate_envelope_compat(::AbstractDistance, p::AbstractProfile)
    p.R isa Real || error(
        "RadialEnvelope: scalar distance metrics (Euclidean / Axial / " *
        "Perpendicular) require a scalar profile radius, but profile carries " *
        "R=$(p.R)::$(typeof(p.R)). Use a scalar SinSquareProfile(R) / " *
        "SinPowerProfile{N}(R) / CosineRampProfile(R, edge), or pair the " *
        "per-axis profile with AxisProductDistance().",
    )
    return nothing
end

"""
    site_envelope(env::RadialEnvelope, lat, k::Int) -> Float64

Combined evaluation: `profile_value(env.profile, distance_at(env.distance, lat, k, r_c))`,
where `r_c = center_position(env.center, lat)`. The implementation
lives in `LatticeCoreExt` because it touches `position(lat, k)`.
"""
function site_envelope end

# `center_position` and `distance_at` are declared (no methods) so that
# the extension can add lattice-bound methods without piracy.

"""
    center_position(center::AbstractCenter, lat) -> SVector{D, T}

Resolve the envelope center to a concrete real-space vector. Methods
live in `LatticeCoreExt`.
"""
function center_position end

"""
    distance_at(dist::AbstractDistance, lat, k::Int, r_c) -> Real or NTuple

Evaluate the distance metric at lattice site `k`. Defers to
`distance_at_position(dist, position(lat, k), r_c)`; method lives in
`LatticeCoreExt`.
"""
function distance_at end

# ---------------------------------------------------------------------
# Extension-missing fallbacks
# ---------------------------------------------------------------------
# When the `LatticeCoreExt` extension is not loaded, the lattice-bound
# methods (`center_position(c, lat)`, `distance_at(dist, lat, k, r_c)`,
# `site_envelope`, `site_weight(env, lat, k)`, `bond_weight(env, lat,
# i, j)`) have no methods. Without the fallbacks below, calling them
# yields an opaque `MethodError` with no hint that the user needs to
# `using LatticeCore`. Each fallback dispatches on `Any` for the
# lattice argument; the extension's `AbstractLattice` methods are
# strictly more specific and take precedence when loaded.

"""
    _missing_lat_msg(sig, lat) -> String

Build the error string for an extension-missing fallback. Names the
function signature, reports `typeof(lat)`, and instructs the user to
load `LatticeCore`. Covers both possible diagnoses: the caller may
have forgotten `using LatticeCore`, *or* the extension is already
loaded but they passed a non-`AbstractLattice` value where one was
expected.
"""
function _missing_lat_msg(sig::AbstractString, lat)
    return "$(sig): no method for lat::$(typeof(lat)). This function is " *
           "supplied by the `LatticeCoreExt` package extension and only " *
           "defines methods on `LatticeCore.AbstractLattice`. If you have " *
           "not yet loaded LatticeCore, add it to your project and " *
           "`using LatticeCore`. If LatticeCore is loaded, check the " *
           "`lat` argument — it must be an `AbstractLattice`."
end

function center_position(::AbstractCenter, lat)
    return error(_missing_lat_msg("center_position(::AbstractCenter, lat)", lat))
end

function distance_at(::AbstractDistance, lat, ::Int, _r_c)
    return error(_missing_lat_msg("distance_at(::AbstractDistance, lat, k, r_c)", lat))
end

function site_envelope(::RadialEnvelope, lat, ::Int)
    return error(_missing_lat_msg("site_envelope(::RadialEnvelope, lat, k)", lat))
end

function site_weight(::AbstractModulationND, lat, ::Int)
    return error(_missing_lat_msg("site_weight(::AbstractModulationND, lat, k)", lat))
end

function bond_weight(::AbstractModulationND, lat, ::Int, ::Int)
    return error(_missing_lat_msg("bond_weight(::AbstractModulationND, lat, i, j)", lat))
end
