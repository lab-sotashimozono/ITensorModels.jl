using Random: AbstractRNG, Xoshiro

"""
    RandomTFIM(J, h, site)
    RandomTFIM(; J, h, site = SiteType("S=1/2"))

Open transverse-field Ising chain with SITE-DEPENDENT couplings,

    H = -Σ_{i=1}^{L-1} J_i Z_i Z_{i+1} - Σ_{i=1}^{L} h_i X_i

with `length(J) == L - 1` and `length(h) == L`, and `Z`, `X` the local operators
of `site` exactly as for [`TFIM`](@ref) — same family, same default, so constant
couplings reproduce `TFIM` term for term.

Couplings belong to the chain's ORDINAL bonds and sites, not to MPS positions:
[`local_ham_terms`](@ref) reads `J[k]` for the `k`-th consecutive pair of
`phys_sites` whatever positions those are, so an offset or interleaved layout
carries the same realisation.

Draw one from a seed with [`random_tfim`](@ref), and move its disorder strength
with [`rescale_disorder`](@ref).
"""
struct RandomTFIM <: AbstractLatticeModel
    J::Vector{Float64}
    h::Vector{Float64}
    site::SiteType
    function RandomTFIM(J, h, site)
        length(h) >= 2 || error("RandomTFIM: need at least 2 sites, got $(length(h))")
        length(J) == length(h) - 1 || error(
            "RandomTFIM: an open chain of $(length(h)) sites has $(length(h) - 1) " *
            "bonds, got $(length(J)); a length-$(length(h)) vector would be the " *
            "periodic chain",
        )
        return new(collect(Float64, J), collect(Float64, h), site)
    end
end

function RandomTFIM(; J, h, site=SiteType("S=1/2"))
    return RandomTFIM(J, h, site)
end

"""
    random_tfim(; L, seed, D = 1.0, site = SiteType("S=1/2")) -> RandomTFIM

Draw a [`RandomTFIM`](@ref) of `L` sites with `J_i = u_i^D` and `h_i = v_i^D`,
`u, v` independent and uniform on `(0, 1)`.

That is the disorder-strength parameterisation used by Xavier, Hoyos and
Miranda, Phys. Rev. B 98, 195115 (2018): the density is
`P(J) = J^{1/D - 1} / D` on `(0, 1)`, so `D = 1` is the uniform ensemble and
larger `D` widens the spread of `log J` without moving its shape. `J` and `h`
are drawn from the same distribution, so `[ln J] = [ln h]` and the chain sits on
the self-dual critical line at every `D`.

`seed` fixes the couplings, and the couplings are the whole model, so the same
seed gives the same chain to every representation built from it. An `Integer`
seed must be non-negative (`Xoshiro` rejects negatives on Julia 1.10). Passing an
`AbstractRNG` instead ADVANCES it by `2L - 1` draws, as any sampling function
does.
"""
function random_tfim(;
    L::Int, seed::Union{Integer,AbstractRNG}, D::Real=1.0, site::SiteType=SiteType("S=1/2")
)
    L >= 2 || error("random_tfim: need at least 2 sites, got L = $L")
    D > 0 || error("random_tfim: disorder strength D must be positive, got $D")
    seed isa Integer &&
        seed < 0 &&
        error(
            "random_tfim: an integer seed must be non-negative, got $seed " *
            "(Xoshiro rejects negatives on Julia 1.10)",
        )
    rng = seed isa AbstractRNG ? seed : Xoshiro(seed)
    return RandomTFIM(rand(rng, L - 1) .^ D, rand(rng, L) .^ D, site)
end

"""
    rescale_disorder(m::RandomTFIM, α) -> RandomTFIM

`J_i^α`, `h_i^α` — the same realisation at disorder strength `αD`.

Exact for a chain from [`random_tfim`](@ref), where `J = u^D` and therefore
`J^α = u^{αD}`: the couplings keep their order and their draw, and only the
spread of their logarithms is scaled. `α = 0` is the clean chain `J = h = 1`,
`α = 1` is `m`, and `[ln J]` and `[ln h]` scale together so the chain stays
critical throughout. The continuation step of adaptive DMRG (Xavier, Hoyos and
Miranda, Phys. Rev. B 98, 195115 (2018)).

On a hand-built `RandomTFIM` this is still `J^α`; whether that is a disorder
strength depends on how the couplings were made.
"""
function rescale_disorder(m::RandomTFIM, α::Real)
    α >= 0 || error("rescale_disorder: α must be non-negative, got $α")
    all(≥(0), m.J) && all(≥(0), m.h) || error(
        "rescale_disorder: needs non-negative couplings; got $(count(<(0), m.J)) " *
        "negative bonds and $(count(<(0), m.h)) negative fields",
    )
    return RandomTFIM(m.J .^ α, m.h .^ α, m.site)
end

site_type(m::RandomTFIM) = m.site

function onsite_observable_op(m::RandomTFIM, name::Symbol)
    name === :sx && return ising_x_op(m.site)
    name === :sz && return ising_z_op(m.site)
    name === :sy && return _tfim_y_op(m.site)
    return error("RandomTFIM: unsupported onsite observable $name on site $(m.site)")
end

"""
    local_ham_terms(m::RandomTFIM, phys_sites; boundary = :bulk_half_edge)

The chain's local pieces, with `J[k]` on the `k`-th consecutive pair of
`phys_sites` and `h[k]` on the `k`-th.

Overridden because the generic implementation passes `bond_term` the two MPS
POSITIONS and drops the ordinal, which for site-dependent couplings is the
difference between the realisation the seed drew and a shifted one. `phys_sites`
must therefore be the whole chain: a window of it names sites the couplings do
not describe.
"""
function local_ham_terms(m::RandomTFIM, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    N == length(m.h) || error(
        "RandomTFIM: the model carries couplings for $(length(m.h)) sites but was " *
        "given $N phys sites; a site-dependent chain has no meaning on a window of " *
        "itself",
    )
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    terms = OpSum[]
    for k in 1:(N - 1)
        H = OpSum()
        H += -m.J[k], zop, phys[k], zop, phys[k + 1]
        H += -m.h[k] / 2, xop, phys[k]
        H += -m.h[k + 1] / 2, xop, phys[k + 1]
        push!(terms, H)
    end
    if boundary === :full
        for (k, pos) in ((1, phys[1]), (N, phys[end]))
            H = OpSum()
            H += -m.h[k] / 2, xop, pos
            push!(terms, H)
        end
    elseif boundary !== :bulk_half_edge
        error("local_ham_terms: unknown boundary $boundary")
    end
    return terms
end

# `bond_term` and the split protocol are handed MPS positions with no ordinal, so
# they can only read the couplings by treating position AS ordinal — true on a
# plain `1:L` chain and nowhere else. `local_ham_terms` above is the general path;
# these stay for direct callers (bond-resolved energy density) and for
# `ModulatedModel`, and refuse anything outside `1:L`.
function _bond_index(m::RandomTFIM, i::Int, j::Int)
    j == i + 1 || error(
        "RandomTFIM: bond_term reads J by the bond's left site, so it needs a " *
        "nearest-neighbour ascending pair of a 1:L chain (got i=$i, j=$j); use " *
        "local_ham_terms for any other layout",
    )
    1 <= i <= length(m.h) - 1 ||
        error("RandomTFIM: bond $i out of range 1:$(length(m.h) - 1)")
    return i
end

function _site_index(m::RandomTFIM, k::Int)
    1 <= k <= length(m.h) || error("RandomTFIM: site $k out of range 1:$(length(m.h))")
    return k
end

function bond_term(m::RandomTFIM, i::Int, j::Int)
    b = _bond_index(m, i, j)
    zop = ising_z_op(m.site)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.J[b], zop, i, zop, j
    opsum += -m.h[_site_index(m, i)] / 2, xop, i
    opsum += -m.h[_site_index(m, j)] / 2, xop, j
    return opsum
end

function boundary_patch(m::RandomTFIM, i::Int)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h[_site_index(m, i)] / 2, xop, i
    return opsum
end

function bond_coupling_term(m::RandomTFIM, i::Int, j::Int)
    zop = ising_z_op(m.site)
    opsum = OpSum()
    opsum += -m.J[_bond_index(m, i, j)], zop, i, zop, j
    return opsum
end

function onsite_term(m::RandomTFIM, k::Int)
    xop = ising_x_op(m.site)
    opsum = OpSum()
    opsum += -m.h[_site_index(m, k)], xop, k
    return opsum
end
