"""
    bond_term(model::AbstractLatticeModel, i::Int, j::Int) -> OpSum

Local piece of the Hamiltonian associated with the bond between physical
positions `i` and `j`. Implementations should distribute on-site terms
**symmetrically with half weight on each endpoint** so that summing
`bond_term` over every consecutive pair in a chain reproduces the
`:bulk_half_edge` convention (each interior phys site accumulates the
full on-site weight from two adjacent bonds, while the two end sites
retain half weight).

This is the local energy density: `⟨ψ|MPO(bond_term(m, i, j), sites)|ψ⟩`
is the energy on bond `(i, j)` and nearby sites.
"""
function bond_term end

"""
    boundary_patch(model::AbstractLatticeModel, i::Int) -> OpSum

Extra on-site contribution added at the two boundary phys sites when
`build_opsum` is called with `boundary = :full`. Default: empty
`OpSum()`. Override to supply the missing half-weight on-site terms so
that summing `bond_term`s over all bonds plus `boundary_patch`es at both
ends reproduces the plain-chain Hamiltonian.
"""
boundary_patch(::AbstractLatticeModel, ::Int) = OpSum()

"""
    local_ham_terms(model, phys_sites; boundary=:bulk_half_edge) -> Vector{OpSum}

Ordered list of local Hamiltonian pieces whose sum is the full MPO.
The first `length(phys_sites)-1` entries are `bond_term(model, phys_sites[k], phys_sites[k+1])`;
for `boundary = :full` two extra `boundary_patch` entries are appended.

Useful directly for local observables (bond-resolved energy density,
per-site magnetisation from the relevant piece, …).
"""
function local_ham_terms(
    m::AbstractLatticeModel, phys_sites; boundary::Symbol=:bulk_half_edge
)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for k in 1:(N - 1)
        push!(terms, bond_term(m, phys[k], phys[k + 1]))
    end
    if boundary === :full && N >= 1
        push!(terms, boundary_patch(m, phys[1]))
        N >= 2 && push!(terms, boundary_patch(m, phys[end]))
    elseif boundary === :bulk_half_edge
        # no boundary patches
    else
        error("local_ham_terms: unknown boundary $boundary")
    end
    return terms
end

"""
    build_opsum(model, sites; phys_sites, boundary=:bulk_half_edge) -> OpSum

Compose the full Hamiltonian `OpSum` by summing
[`local_ham_terms`](@ref). Every concrete `AbstractLatticeModel` gets
this for free as long as it defines [`bond_term`](@ref) (and, if it
wants `:full` support, [`boundary_patch`](@ref)).
"""
function build_opsum(
    m::AbstractLatticeModel,
    sites;
    phys_sites=findall(i -> hastags(i, PhysSite), sites),
    boundary::Symbol=:bulk_half_edge,
)
    opsum = OpSum()
    for term in local_ham_terms(m, phys_sites; boundary)
        opsum += term
    end
    return opsum
end

# ---------------------------------------------------------------------
# Split protocol (opt-in)
# ---------------------------------------------------------------------
#
# The plain `bond_term` returns an `OpSum` that mixes bond couplings
# with half-weight on-site terms; once aggregated those parts cannot
# be separated. Models that want to support per-bond / per-site
# modulation (Sine-Square Deformation, Smooth Boundary, Hotta-Shibata
# sin^N, custom envelopes, …) opt in by implementing the two methods
# below. `ModulatedModel` consumes the split protocol; the existing
# `bond_term` / `boundary_patch` of base models stays untouched.

"""
    bond_coupling_term(model, i, j) -> OpSum

Pure two-site coupling on bond `(i, j)` with **no** on-site weight.
For TFIM this is `-J Zᵢ Zⱼ`. Models override this to participate in the
split protocol consumed by [`ModulatedModel`](@ref).
"""
function bond_coupling_term end

"""
    bond_hopping_term(model, i, j) -> OpSum

The FORWARD hopping on bond `(i, j)` alone: the part of [`bond_coupling_term`](@ref) that a
Peierls phase multiplies, without its conjugate and without any density-density piece.

A Trotter gate for a driven chain needs it separately, because the phase is time dependent
and the amplitude is not: a consumer that has this can scale a cached operator instead of
rebuilding one per step, and does not have to retype the model's own amplitudes to get it.

Defined for the models that carry a vector potential, where the split has meaning.
"""
function bond_hopping_term end

"""
    onsite_term(model, k) -> OpSum

Pure on-site term at site `k` with **full** weight (not halved). For
TFIM this is `-h Xₖ`. Models override this to participate in the
split protocol consumed by [`ModulatedModel`](@ref).
"""
function onsite_term end
