module LatticeCoreExt

using ITensors: OpSum, hastags
using ITensorModels
using ITensorModels:
    AbstractLatticeModel,
    LatticeModel,
    ModulatedLatticeModel,
    bond_term,
    bond_coupling_term,
    onsite_term,
    local_ham_terms,
    boundary_patch,
    AbstractModulationND,
    RadialEnvelope,
    AbstractCenter,
    GeometricCenter,
    BoundingBoxCenter,
    ExplicitCenter,
    AbstractDistance,
    distance_at_position,
    profile_value
using ITensorSiteKit: PhysSite
using LatticeCore: AbstractLattice, position, bonds, num_sites

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"""
    _ordering(m::LatticeModel)

Resolve the lattice-site → MPS-position map. An empty `m.ordering`
selects the natural order `1:num_sites(m.lattice)`.
"""
function _ordering(m::LatticeModel)
    isempty(m.ordering) ? collect(1:num_sites(m.lattice)) : m.ordering
end

"""
    _lookup_bond_model(m::LatticeModel, bt::Symbol)

Fetch the sub-model for `Bond.type == bt`, falling back to `:nearest`
if the specific tag is not registered. This lets a uniform-couplings
dict `Dict(:nearest => Heisenberg1D(...))` cover lattices whose bond
types are labelled differently (`:type_1`, `:type_2`, etc.) as long as
the user passes a dict keyed on the actual labels they want to match.
"""
function _lookup_bond_model(m::LatticeModel, bt::Symbol)
    haskey(m.bond_models, bt) && return m.bond_models[bt]
    haskey(m.bond_models, :nearest) && return m.bond_models[:nearest]
    return error(
        "LatticeModel: no bond_model registered for Bond.type = $bt. " *
        "Known keys: $(collect(keys(m.bond_models))).",
    )
end

# ---------------------------------------------------------------------
# local_ham_terms / build_opsum
# ---------------------------------------------------------------------

"""
    local_ham_terms(m::LatticeModel{<:AbstractLattice}, phys_sites; boundary)

Iterate the lattice's bonds; for each bond emit the corresponding
sub-model's `bond_term` on the MPS positions obtained from `_ordering`.
Single-site boundary patches are **not** emitted — 2D / graph lattices
don't have a canonical boundary in the `:bulk_half_edge` sense, so each
bond term carries the full on-site weight already (this matches the
behaviour of `Heisenberg1D` / `XXZ1D` whose on-site fields are zero).

`phys_sites` is ignored: the lattice itself selects which positions
host physical DOF via its bond connectivity. `boundary` is accepted
for interface compatibility but currently must be `:full`.
"""
function ITensorModels.local_ham_terms(
    m::LatticeModel{<:AbstractLattice}, phys_sites; boundary::Symbol=:full
)
    boundary === :full || error(
        "LatticeModel currently supports only boundary = :full " *
        "(got $boundary). Open a `:bulk_half_edge` variant when a " *
        "ChainConfig that needs it lands.",
    )
    ord = _ordering(m)
    terms = OpSum[]
    for b in bonds(m.lattice)
        submodel = _lookup_bond_model(m, b.type)
        push!(terms, bond_term(submodel, ord[b.i], ord[b.j]))
    end
    return terms
end

"""
    build_opsum(m::LatticeModel{<:AbstractLattice}, sites;
                phys_sites=nothing, boundary=:full)

Compose the full `OpSum` by summing every bond term produced by
[`local_ham_terms`](@ref). `phys_sites` is ignored (the lattice graph
provides connectivity).
"""
function ITensorModels.build_opsum(
    m::LatticeModel{<:AbstractLattice}, sites; phys_sites=nothing, boundary::Symbol=:full
)
    opsum = OpSum()
    for t in ITensorModels.local_ham_terms(m, phys_sites; boundary)
        opsum += t
    end
    return opsum
end

# ---------------------------------------------------------------------
# ND modulation envelopes — lattice-bound evaluation
# ---------------------------------------------------------------------

"""
    center_position(::GeometricCenter, lat::AbstractLattice)

Geometric mean of all site positions, `r_c = (1/N) Σ_k r_k`. The
returned value is the same type as `position(lat, 1)` (typically an
`SVector{D, T}`), so downstream arithmetic stays type-stable on the
StaticArrays fast path provided by `LatticeCore`.
"""
function ITensorModels.center_position(::GeometricCenter, lat::AbstractLattice)
    N = num_sites(lat)
    N >= 1 || error("center_position: lattice has zero sites")
    acc = position(lat, 1)
    for k in 2:N
        acc = acc + position(lat, k)
    end
    return acc / N
end

"""
    center_position(::BoundingBoxCenter, lat::AbstractLattice)

Midpoint of the axis-aligned bounding box, `(min(r_k) + max(r_k)) / 2`
per axis.
"""
function ITensorModels.center_position(::BoundingBoxCenter, lat::AbstractLattice)
    N = num_sites(lat)
    N >= 1 || error("center_position: lattice has zero sites")
    lo = position(lat, 1)
    hi = lo
    for k in 2:N
        p = position(lat, k)
        lo = min.(lo, p)
        hi = max.(hi, p)
    end
    return (lo + hi) / 2
end

"""
    center_position(c::ExplicitCenter, lat::AbstractLattice)

Return the user-supplied center vector. Length must match
`dimension(lat)`.
"""
function ITensorModels.center_position(c::ExplicitCenter, lat::AbstractLattice)
    length(c.r) == length(position(lat, 1)) || error(
        "ExplicitCenter: r has length $(length(c.r)) but lattice has " *
        "dimension $(length(position(lat, 1)))",
    )
    return c.r
end

"""
    distance_at(dist::AbstractDistance, lat::AbstractLattice, k::Int, r_c)

Evaluate the distance metric at lattice site `k`. Defers to
`distance_at_position(dist, position(lat, k), r_c)`.
"""
function ITensorModels.distance_at(
    dist::AbstractDistance, lat::AbstractLattice, k::Int, r_c
)
    return distance_at_position(dist, position(lat, k), r_c)
end

"""
    site_envelope(env::RadialEnvelope, lat::AbstractLattice, k::Int)

`profile_value(env.profile, distance_at(env.distance, lat, k, r_c))`
with `r_c = center_position(env.center, lat)`.
"""
function ITensorModels.site_envelope(env::RadialEnvelope, lat::AbstractLattice, k::Int)
    r_c = ITensorModels.center_position(env.center, lat)
    d = ITensorModels.distance_at(env.distance, lat, k, r_c)
    return profile_value(env.profile, d)
end

"""
    site_weight(env::AbstractModulationND, lat::AbstractLattice, k::Int)

ND `site_weight`: equals `site_envelope(env, lat, k)`. This is the
ND-signature companion to the 1D `site_weight(mod, i::Int, L::Int)`.
"""
function ITensorModels.site_weight(env::RadialEnvelope, lat::AbstractLattice, k::Int)
    return ITensorModels.site_envelope(env, lat, k)
end

"""
    bond_weight(env::AbstractModulationND, lat::AbstractLattice, i::Int, j::Int)

ND `bond_weight`: midpoint evaluation
`f((r_i + r_j) / 2)`. Matches the 1D convention
`bond_weight(SSD, i, L) = sin²(π i / L)` (sites at half-integer
positions, bond midpoints at integers).
"""
function ITensorModels.bond_weight(
    env::RadialEnvelope, lat::AbstractLattice, i::Int, j::Int
)
    r_c = ITensorModels.center_position(env.center, lat)
    r_mid = (position(lat, i) + position(lat, j)) / 2
    d = distance_at_position(env.distance, r_mid, r_c)
    return profile_value(env.profile, d)
end

# ---------------------------------------------------------------------
# ModulatedLatticeModel local_ham_terms
# ---------------------------------------------------------------------

"""
    _onsite_submodel_for(m::LatticeModel, k::Int)

Return the submodel whose `onsite_term` is used at lattice site `k`.
Implementation: pick the bond model attached to any bond touching `k`.
All bundled models (`TFIM`, `TFIML`, `XXZ1D`, `Heisenberg1D`) have
site-uniform on-site terms, so this is well-defined; a model with
genuinely per-site fields would need a separate accessor.
"""
function _onsite_submodel_for(m::LatticeModel, k::Int)
    for b in bonds(m.lattice)
        if b.i == k || b.j == k
            return _lookup_bond_model(m, b.type)
        end
    end
    return error(
        "ModulatedLatticeModel: no bond touches site $k; cannot resolve " *
        "onsite_term submodel.",
    )
end

"""
    local_ham_terms(m::ModulatedLatticeModel{<:LatticeModel{<:AbstractLattice}}, _; boundary)

ND modulation pipeline. Iterate the lattice'`s bonds, weighting each
`bond_coupling_term` by `bond_weight(env, lat, i, j)` (midpoint
evaluation); then iterate sites, weighting each `onsite_term` by
`site_weight(env, lat, k)`. Bond and on-site contributions are emitted
as separate `OpSum`s — no half-distribution, no `boundary_patch`.

`phys_sites` is ignored (the lattice graph provides connectivity).
`boundary` is accepted for interface compatibility but currently must
be `:full`.
"""
function ITensorModels.local_ham_terms(
    m::ModulatedLatticeModel{<:LatticeModel{<:AbstractLattice}}, phys_sites;
    boundary::Symbol=:full,
)
    boundary === :full || error(
        "ModulatedLatticeModel currently supports only boundary = :full " *
        "(got $boundary).",
    )
    base = m.base
    lat = base.lattice
    ord = _ordering(base)
    env = m.envelope
    terms = OpSum[]
    for b in bonds(lat)
        sub = _lookup_bond_model(base, b.type)
        fb = ITensorModels.bond_weight(env, lat, b.i, b.j)
        push!(terms, fb * bond_coupling_term(sub, ord[b.i], ord[b.j]))
    end
    for k in 1:num_sites(lat)
        sub = _onsite_submodel_for(base, k)
        fk = ITensorModels.site_weight(env, lat, k)
        push!(terms, fk * onsite_term(sub, ord[k]))
    end
    return terms
end

"""
    build_opsum(m::ModulatedLatticeModel{<:LatticeModel{<:AbstractLattice}}, sites;
                phys_sites=nothing, boundary=:full)

Sum every term produced by [`local_ham_terms`](@ref) for the modulated
lattice model into the full `OpSum`.
"""
function ITensorModels.build_opsum(
    m::ModulatedLatticeModel{<:LatticeModel{<:AbstractLattice}}, sites;
    phys_sites=nothing, boundary::Symbol=:full,
)
    opsum = OpSum()
    for t in ITensorModels.local_ham_terms(m, phys_sites; boundary)
        opsum += t
    end
    return opsum
end

end # module LatticeCoreExt
