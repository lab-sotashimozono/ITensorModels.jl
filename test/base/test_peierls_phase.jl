using ITensorModels
using ITensors
using ITensorMPS
using AbstractQAtlas: VectorPotential, peierls_phase
using LinearAlgebra: norm
using Test

# The bond `i → j` as the abstract layer defines it: amplitude `-t`, phase `A * d` with `d` the
# model's declared `bond_displacement`, contracted by `AbstractQAtlas.peierls_phase`. Written
# out here rather than taken from the model, so the two sides stay independent.
function _abstract_bond(m::RiceMele1D, i::Int, j::Int)
    t = isodd(i) ? m.v : m.w
    phase = peierls_phase(VectorPotential(m.A), (bond_displacement(m),))
    H = OpSum()
    H += (-t * cis(-phase), "Cdag", i, "C", j)
    H += (-t * cis(phase), "Cdag", j, "C", i)
    iszero(m.V) || (H += (m.V, "N", i, "N", j))
    return H
end

function _abstract_bond(m::RiceMeleHubbard1D, i::Int, j::Int)
    t = isodd(i) ? m.v : m.w
    phase = peierls_phase(VectorPotential(m.A), (bond_displacement(m),))
    H = OpSum()
    for (dag, ann) in (("Cdagup", "Cup"), ("Cdagdn", "Cdn"))
        H += (-t * cis(-phase), dag, i, ann, j)
        H += (-t * cis(phase), dag, j, ann, i)
    end
    return H
end

@testset "bond_displacement is one site spacing, and takes no site indices" begin
    for m in (RiceMele1D(; A=0.35), RiceMeleHubbard1D(; A=0.35))
        @test bond_displacement(m) == 1.0
        # Deliberately NOT a function of `(i, j)`: on the `LatticeCore` path those are MPS
        # positions from a user-supplied ordering, so their difference is not a displacement.
        @test !hasmethod(bond_displacement, Tuple{typeof(m),Int,Int})
    end
end

@testset "the models' Peierls phase is AbstractQAtlas.peierls_phase" begin
    # This is a two-way pin, and it has to be: with the displacement equal to 1, a model that
    # dropped `bond_displacement` and wrote `cis(-m.A)` builds the SAME operator, so "the code
    # multiplies by d" is not a testable claim. What is testable is that the DECLARED unit and
    # the operator agree — which fails if either one moves. The control below runs both moves.
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9, A=0.35), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0, A=0.35), "Electron"),
    )
        sites = siteinds(st, 3)
        for (i, j) in ((1, 2), (2, 3))
            got = prod(MPO(bond_coupling_term(m, i, j), sites))
            want = prod(MPO(_abstract_bond(m, i, j), sites))
            @test norm(got - want) < 1.0e-12 * max(norm(want), 1.0)
        end
    end
end

@testset "the comparison above can fail, from either side" begin
    # The reference above reads `bond_displacement` too, so it cannot see the DECLARED unit
    # being wrong — only a mismatch. That half is covered by the value pin in the first
    # testset. What is left to show is that the operator's phase is pinned to a number at all,
    # rather than to whatever the reference happens to compute: two neighbouring conventions
    # have to be rejected.
    m = RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9, A=0.35)
    sites = siteinds("Fermion", 3)
    ref(phase) =
        let H = OpSum()
            H += (-m.v * cis(-phase), "Cdag", 1, "C", 2)
            H += (-m.v * cis(phase), "Cdag", 2, "C", 1)
            H += (m.V, "N", 1, "N", 2)
            prod(MPO(H, sites))
        end
    got = prod(MPO(bond_coupling_term(m, 1, 2), sites))
    @test norm(got - ref(m.A * bond_displacement(m))) < 1.0e-12   # the declared unit
    @test norm(got - ref(m.A * 0.5)) > 0.1                        # a source using the unit cell
    @test norm(got - ref(m.A * 2.0)) > 0.1                        # ...or twice the site spacing
end

@testset "a zero field builds a real operator" begin
    # The `iszero(m.A)` shortcut in `_hop_amplitudes` exists to keep the undriven model real;
    # routing the phase through `bond_displacement` must not have cost that.
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0), "Electron"),
    )
        @test eltype(prod(MPO(bond_coupling_term(m, 1, 2), siteinds(st, 2)))) <: Real
    end
end
