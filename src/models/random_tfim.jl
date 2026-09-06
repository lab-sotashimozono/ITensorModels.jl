using Random: AbstractRNG, Xoshiro

"""
    RandomTFIM(J, h, site)
    RandomTFIM(; J, h, site = SiteType("Qubit"))

Open transverse-field Ising chain with SITE-DEPENDENT couplings,

    H = -Σ_{i=1}^{L-1} J_i Z_i Z_{i+1} - Σ_{i=1}^{L} h_i X_i

with `length(J) == L - 1` and `length(h) == L`, and `Z`, `X` the local operators
of `site` exactly as for [`TFIM`](@ref). The axis convention is `TFIM`'s, so the
two are the same family and a disordered chain reduces to a clean one when the
couplings are constant.

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

function RandomTFIM(; J, h, site=SiteType("Qubit"))
    return RandomTFIM(J, h, site)
end

"""
    random_tfim(; L, seed, D = 1.0, Omega = 1.0, site = SiteType("Qubit")) -> RandomTFIM

Draw a [`RandomTFIM`](@ref) of `L` sites:

    J_i = Ω u_i^D,   h_i = Ω v_i^D,   u, v ~ U(0, 1),

i.e. `P(J) = (D/Ω)(Ω/J)^{1-1/D}`, the disorder-strength parameterisation of
Xavier, Hoyos and Miranda, Phys. Rev. B 98, 195115 (2018). `D = 1` is the
uniform ensemble.
Because `J` and `h` share a distribution, `[ln J] = [ln h]` and the chain sits on
the self-dual critical line for every `D`.

`seed` may be an integer or an `AbstractRNG`. It fixes the couplings and nothing
else, so the same seed gives the same chain to every downstream representation:
the MPO built from this model and a Jordan-Wigner treatment of the same `J`, `h`
are the same Hamiltonian, not equivalent ones.
"""
function random_tfim(;
    L::Int,
    seed::Union{Integer,AbstractRNG},
    D::Real=1.0,
    Omega::Real=1.0,
    site::SiteType=SiteType("Qubit"),
)
    L >= 2 || error("random_tfim: need at least 2 sites, got L = $L")
    D > 0 || error("random_tfim: disorder strength D must be positive, got $D")
    rng = seed isa AbstractRNG ? seed : Xoshiro(seed)
    return RandomTFIM(Omega .* rand(rng, L - 1) .^ D, Omega .* rand(rng, L) .^ D, site)
end

"""
    rescale_disorder(m::RandomTFIM, α) -> RandomTFIM

`J_i^α`, `h_i^α` — the same realisation at disorder strength `αD`.

The continuation step of adaptive DMRG (Xavier, Hoyos and Miranda,
Phys. Rev. B 98, 195115 (2018)): the couplings keep their order and their draw,
only the spread of their logarithms is scaled. `α = 0` is the clean
chain `J = h = Ω`, `α = 1` is `m`, and criticality is preserved throughout
because `[ln J]` and `[ln h]` scale together.
"""
function rescale_disorder(m::RandomTFIM, α::Real)
    α >= 0 || error("rescale_disorder: α must be non-negative, got $α")
    return RandomTFIM(m.J .^ α, m.h .^ α, m.site)
end

site_type(m::RandomTFIM) = m.site

function onsite_observable_op(m::RandomTFIM, name::Symbol)
    return onsite_observable_op(TFIM(; J=1.0, h=1.0, site=m.site), name)
end

# The coupling belongs to a lattice bond, so `bond_term` reads J by the bond's
# left site: only a contiguous 1:L chain of phys sites is meaningful here.
function _bond_index(m::RandomTFIM, i::Int, j::Int)
    j == i + 1 || error(
        "RandomTFIM: site-dependent couplings are defined on nearest-neighbour " *
        "bonds of a 1:L chain (got i=$i, j=$j)",
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
