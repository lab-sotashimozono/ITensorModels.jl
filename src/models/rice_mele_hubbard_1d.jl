using ITensors: SiteType, @SiteType_str

"""
    RiceMeleHubbard1D(; v=1.0, w=1.0, Δ=0.0, U=0.0, A=0.0, site=SiteType("Electron"))

Rice-Mele chain with on-site Hubbard repulsion, for spinful fermions:

    H = -v Σ_{i odd,σ}  (c†_iσ c_{i+1,σ} + h.c.)
        -w Σ_{i even,σ} (c†_iσ c_{i+1,σ} + h.c.)
        + Δ Σ_i (-1)^{i+1} n_i
        + U Σ_i n_{i↑} n_{i↓}

`A` is a uniform vector potential, entering as a Peierls phase `e^{∓iA}` on the two
hopping directions. MEASURED, by Fourier-transforming the ring: the convention here sends
`H_k -> H_{k-A}` (`1.5e-15` against `2.6` for the other sign, per k — the spectrum alone
cannot see it, since `E(k)` is even). On an OPEN chain it is pure gauge and moves no
energy, measured at `1e-15` over `A ∈ {0.3, 1.0, 2.5}`; it is the ring, or the time
derivative, that makes it physical.

Dimerised hopping (`v` intracell, `w` intercell) plus a staggered potential `Δ`
that opens a gap the SSH chain's chiral symmetry forbids: `√((|v|−|w|)² + Δ²)`,
which never closes. At `U = 0` the two spin species decouple and each fills the
lower band at half filling, so the energy is twice the spinless Rice-Mele value —
`test/base/test_rice_mele_hubbard_1d_vs_qatlas.jl` uses that against QAtlas.

`Δ` is the full sublattice offset, not half of it: site 1 sits at `+Δ`.
"""
Base.@kwdef struct RiceMeleHubbard1D <: AbstractLatticeModel
    v::Float64 = 1.0
    w::Float64 = 1.0
    Δ::Float64 = 0.0
    U::Float64 = 0.0
    A::Float64 = 0.0
    site::SiteType = SiteType("Electron")
end

site_type(m::RiceMeleHubbard1D) = m.site

_hop(m::RiceMeleHubbard1D, i::Int) = isodd(i) ? m.v : m.w

# Long form: a constant one-line body is folded away and records no coverage hit.
function bond_displacement(::RiceMeleHubbard1D)
    return 1.0
end

# Forward and backward hopping amplitudes on bond `i`. Real when there is no field, so the
# field-free model still builds a real MPO rather than a complex one carrying zero phase.
function _hop_amplitudes(m::RiceMeleHubbard1D, i::Int)
    t = _hop(m, i)
    iszero(m.A) && return (-t, -t)
    phase = m.A * bond_displacement(m)
    return (-t * cis(-phase), -t * cis(phase))
end
_stagger(m::RiceMeleHubbard1D, k::Int) = isodd(k) ? m.Δ : -m.Δ

# J = -∂H/∂A. `A` enters only through the Peierls phase, so with fwd = -t e^{-iAd} and
# bwd = -t e^{+iAd} the derivative is ∂fwd/∂A = -i d fwd and ∂bwd/∂A = +i d bwd. Reusing the
# model's own amplitudes keeps the phase and the length unit single-sourced.
function bond_current_term(m::RiceMeleHubbard1D, i::Int, j::Int)
    d = bond_displacement(m)
    fwd, bwd = _hop_amplitudes(m, i)
    J = OpSum()
    for (dag, ann) in (("Cdagup", "Cup"), ("Cdagdn", "Cdn"))
        J += (im * d * fwd, dag, i, ann, j)
        J += (-im * d * bwd, dag, j, ann, i)
    end
    return J
end

function bond_coupling_term(m::RiceMeleHubbard1D, i::Int, j::Int)
    fwd, bwd = _hop_amplitudes(m, i)
    H = OpSum()
    H += fwd, "Cdagup", i, "Cup", j
    H += bwd, "Cdagup", j, "Cup", i
    H += fwd, "Cdagdn", i, "Cdn", j
    H += bwd, "Cdagdn", j, "Cdn", i
    return H
end

function onsite_term(m::RiceMeleHubbard1D, k::Int)
    H = OpSum()
    H += m.U, "Nupdn", k
    H += _stagger(m, k), "Ntot", k
    return H
end

function bond_term(m::RiceMeleHubbard1D, i::Int, j::Int)
    H = bond_coupling_term(m, i, j)
    H += (m.U / 2), "Nupdn", i
    H += (m.U / 2), "Nupdn", j
    H += _stagger(m, i) / 2, "Ntot", i
    H += _stagger(m, j) / 2, "Ntot", j
    return H
end

function boundary_patch(m::RiceMeleHubbard1D, k::Int)
    H = OpSum()
    H += (m.U / 2), "Nupdn", k
    H += _stagger(m, k) / 2, "Ntot", k
    return H
end

function onsite_observable_op(::RiceMeleHubbard1D, name::Symbol)
    name === :nup && return "Nup"
    name === :ndn && return "Ndn"
    name === :n && return "Ntot"
    name === :nupdn && return "Nupdn"
    name === :sz && return "Sz"
    return error("RiceMeleHubbard1D: unsupported onsite observable $name")
end

# The dimerisation and the staggered potential are read off the PARITY of each site's index,
# and the split protocol is handed a label on the 1D path and an ordinal on the ND one — so
# "which site is this" is not something a bare index answers. SSH1D sidesteps that by leaving
# the split protocol empty and overriding this with positions, which costs it every
# modulation wrapper. Here the two readings are kept identical by construction instead:
# an embedding that would make them differ is refused.
#
# It is not a hypothetical. MEASURED at (v, w, Δ) = (1.0, 0.2, 0.5) on four sites,
# `‖H‖` = 102.0 for `phys_sites = 1:4` against 80.4 for `2:5` — the same chain, shifted by
# one site, with v and w swapped and the sublattices exchanged. SSH1D returns 8.079604 for
# both.
function local_ham_terms(m::RiceMeleHubbard1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    isempty(phys) ||
        (all(==(1), diff(phys)) && isodd(first(phys))) ||
        throw(
            ArgumentError(
                "RiceMeleHubbard1D: phys_sites must be consecutive and start on an odd site, " *
                "got $(first(phys)):$(step_or_gaps(phys)):$(last(phys)). The intracell bond " *
                "`v` is the one leaving an odd site, so any other embedding silently swaps " *
                "v with w and exchanges the sublattices.",
            ),
        )
    return invoke(
        local_ham_terms, Tuple{AbstractLatticeModel,Any}, m, phys_sites; boundary=boundary
    )
end

step_or_gaps(phys) = length(phys) < 2 ? 1 : (allequal(diff(phys)) ? phys[2] - phys[1] : "…")
