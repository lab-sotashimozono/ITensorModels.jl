using ITensors: hastags
using ITensorSiteKit: PhysSite

"""
    onsite_observable_op(model::AbstractLatticeModel, name::Symbol) -> String

ITensors operator string that implements the one-site observable
`name` for `model`, resolved against the model's `site_type`. The
`name` namespace is a small set of physically meaningful symbols
each model chooses to support — `:sx`, `:sy`, `:sz` for spin models,
`:nx`, `:ny` for particle models, etc.

Concrete models override this function per supported `(name, SiteType)`
combination. The generic method throws so that unsupported requests
fail loud rather than silently returning a bogus operator name.
"""
function onsite_observable_op(m::AbstractLatticeModel, name::Symbol)
    return error(
        "onsite_observable_op: model $(typeof(m)) has no observable `$name` on site $(site_type(m))",
    )
end

"""
    build_onsite_observable_opsum(model, sites, name::Symbol;
                                   phys_sites, weights) -> OpSum

Assemble the OpSum `Σ_k weights[k] · op(phys_sites[k])` where `op` is
[`onsite_observable_op`](@ref)`(model, name)`. Default `phys_sites`
picks every `PhysSite`-tagged index in `sites`; default `weights`
gives the unnormalised total (`Σᵢ op_i`).

For the per-site mean pass `weights = fill(1 / length(phys_sites), …)`;
for bulk-only measurements on aux-sandwiched layouts
(e.g. `ThermalMPS.AuxChain`) pass the bulk `phys_sites`
directly.
"""
function build_onsite_observable_opsum(
    m::AbstractLatticeModel,
    sites,
    name::Symbol;
    phys_sites=findall(i -> hastags(i, PhysSite), sites),
    weights=ones(length(phys_sites)),
)
    length(weights) == length(phys_sites) || throw(
        ArgumentError(
            "weights length $(length(weights)) ≠ phys_sites length $(length(phys_sites))",
        ),
    )
    op = onsite_observable_op(m, name)
    opsum = OpSum()
    for (k, i) in enumerate(phys_sites)
        opsum += weights[k], op, i
    end
    return opsum
end

"""
    bond_current_term(model, i, j) -> OpSum

The electric current on bond `(i, j)`: `-∂H/∂A` restricted to that bond, unnormalised.

The Peierls phase is the only place `A` enters, so the derivative acts on the hopping
amplitudes alone and the result is built from exactly the two numbers the Hamiltonian
already uses — [`bond_displacement`](@ref) and the model's own forward/backward amplitudes.
There is no second copy of the phase and no second copy of the length unit, which is the
point: a current written independently of `H` is free to disagree with it about `A`, and
nothing at `A = 0` would notice.

`AbstractQAtlas` owns the relation and its sign, as `ElectricCurrentResponse`
(`j - (-dH_dA)`) hanging off `derivative_edge(ElectricCurrent)`. It cannot supply the
operator — its `peierls_current` seam differentiates a scalar energy — so the operator is
written here and `test/base/test_bond_current.jl` checks it against that relation, with
`∂H/∂A` taken numerically from the model's own Hamiltonian.

Consumers normalise. Ono, *Phys. Rev. Lett.* **135**, 026401 (2025) reports
`⟨J⟩ = -N⁻¹⟨∂H/∂A⟩`; the `N⁻¹` is the caller's, as it is for every other `*_term` here.

Unlike [`onsite_observable_op`](@ref) this DOES carry the Hamiltonian's couplings — being a
derivative of the Hamiltonian is what it is, not an accident of how it is assembled.
"""
function bond_current_term end
