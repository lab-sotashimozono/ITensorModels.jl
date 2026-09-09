using ITensorModels
using ITensors
using ITensorMPS
using AbstractQAtlas: VectorPotential, peierls_phase
using LinearAlgebra: norm
using Test

# The bond as the abstract layer defines it, written out here so the two sides stay separate.
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

@testset "bond_displacement is one site spacing" begin
    for m in (RiceMele1D(; A=0.35), RiceMeleHubbard1D(; A=0.35))
        @test bond_displacement(m) == 1.0
        # No `(i, j)` method: on the `LatticeCore` path those are MPS positions from a
        # user-supplied ordering, so their difference is not a displacement.
        @test !hasmethod(bond_displacement, Tuple{typeof(m),Int,Int})
    end
end

@testset "the models' Peierls phase is AbstractQAtlas.peierls_phase" begin
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

@testset "the comparison above rejects a neighbouring convention" begin
    # The reference shares `bond_displacement`, so it can only see a MISMATCH; the declared
    # value itself is pinned above. What is left is that the phase is pinned to a number.
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
    @test norm(got - ref(m.A * bond_displacement(m))) < 1.0e-12
    @test norm(got - ref(m.A * 0.5)) > 0.1
    @test norm(got - ref(m.A * 2.0)) > 0.1
end

@testset "a zero field builds a real operator" begin
    # `_hop_amplitudes` short-circuits at `A = 0`; routing through `bond_displacement` must
    # not have cost that.
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0), "Electron"),
    )
        @test eltype(prod(MPO(bond_coupling_term(m, 1, 2), siteinds(st, 2)))) <: Real
    end
end
