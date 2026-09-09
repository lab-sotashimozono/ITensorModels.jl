using ITensors: SiteType, @SiteType_str

"""
    RiceMele1D(; v=1.0, w=1.0, Δ=0.0, V=0.0, A=0.0, site=SiteType("Fermion"))

Rice-Mele chain for **spinless** fermions:

    H = -v Σ_{i odd}  (e^{-iA} c†_i c_{i+1} + h.c.)
        -w Σ_{i even} (e^{-iA} c†_i c_{i+1} + h.c.)
        + Δ Σ_i (-1)^{i+1} n_i
        + V Σ_i n_i n_{i+1}

[`SSH1D`](@ref) plus the staggered potential that opens a gap chiral symmetry forbids,
`√((|v|−|w|)² + Δ²)`, which never closes.

The interaction is a NEAREST-NEIGHBOUR repulsion `V`, not the on-site `U` of
[`RiceMeleHubbard1D`](@ref): a spinless site has `n² = n`, so there is no on-site term to
write. That is why this is a separate model rather than the same one on another site type.

`A` is a uniform vector potential entering as a Peierls phase, with the same convention as
[`RiceMeleHubbard1D`](@ref) — `H_k -> H_{k-A}`, measured per k. On an open chain it is pure
gauge and moves no energy.
"""
Base.@kwdef struct RiceMele1D <: AbstractLatticeModel
    v::Float64 = 1.0
    w::Float64 = 1.0
    Δ::Float64 = 0.0
    V::Float64 = 0.0
    A::Float64 = 0.0
    site::SiteType = SiteType("Fermion")
end

site_type(m::RiceMele1D) = m.site

_hop(m::RiceMele1D, i::Int) = isodd(i) ? m.v : m.w

# Sites one unit apart, so `A` is per site spacing. The docstring says why that is a choice
# and what a comparison against a source using another one has to convert.
# Long form on purpose: a one-line method with a constant body is folded away and never
# registers a coverage hit, so the declaration would read as untested.
function bond_displacement(::RiceMele1D)
    return 1.0
end

# Forward and backward hopping amplitudes on the bond from `i` to `j`. Real when there is
# no field, so the field-free model still builds a real MPO rather than a complex one
# carrying zero phase.
function _hop_amplitudes(m::RiceMele1D, i::Int)
    t = _hop(m, i)
    iszero(m.A) && return (-t, -t)
    # Peierls: `c†_i c_j` picks up `exp(-i A d)`. `test/base/test_peierls_phase.jl` pins
    # this against `AbstractQAtlas.peierls_phase`, so the declaration and the operator
    # cannot drift apart in either direction.
    phase = m.A * bond_displacement(m)
    return (-t * cis(-phase), -t * cis(phase))
end
_stagger(m::RiceMele1D, k::Int) = isodd(k) ? m.Δ : -m.Δ

function bond_coupling_term(m::RiceMele1D, i::Int, j::Int)
    fwd, bwd = _hop_amplitudes(m, i)
    H = OpSum()
    H += fwd, "Cdag", i, "C", j
    H += bwd, "Cdag", j, "C", i
    iszero(m.V) || (H += (m.V, "N", i, "N", j))
    return H
end

function onsite_term(m::RiceMele1D, k::Int)
    H = OpSum()
    H += _stagger(m, k), "N", k
    return H
end

function bond_term(m::RiceMele1D, i::Int, j::Int)
    H = bond_coupling_term(m, i, j)
    H += _stagger(m, i) / 2, "N", i
    H += _stagger(m, j) / 2, "N", j
    return H
end

function boundary_patch(m::RiceMele1D, k::Int)
    H = OpSum()
    H += _stagger(m, k) / 2, "N", k
    return H
end

function onsite_observable_op(::RiceMele1D, name::Symbol)
    name === :n && return "N"
    name === :c && return "C"
    name === :cdag && return "Cdag"
    return error("RiceMele1D: unsupported onsite observable $name")
end

# Same reason as `RiceMeleHubbard1D`: the parity of the index is the model, and the split
# protocol is handed a label on the 1D path and an ordinal on the ND one. Keeping the two
# readings identical by construction is cheaper than picking one and being wrong elsewhere.
function local_ham_terms(m::RiceMele1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    isempty(phys) ||
        (all(==(1), diff(phys)) && isodd(first(phys))) ||
        throw(
            ArgumentError(
                "RiceMele1D: phys_sites must be consecutive and start on an odd site, got " *
                "$(first(phys)):$(last(phys)). The intracell bond `v` is the one leaving an " *
                "odd site, so any other embedding silently swaps v with w and exchanges the " *
                "sublattices.",
            ),
        )
    return invoke(
        local_ham_terms, Tuple{AbstractLatticeModel,Any}, m, phys_sites; boundary=boundary
    )
end
