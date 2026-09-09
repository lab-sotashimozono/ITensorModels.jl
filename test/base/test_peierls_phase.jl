using ITensorModels
using ITensors
using ITensorMPS
using AbstractQAtlas: VectorPotential, peierls_phase
using LinearAlgebra: norm
using Test

# The bond `i → j` as the abstract layer defines it: amplitude `-t`, phase `A ⋅ d` with `d` the
# model's own `bond_displacement`, contracted by `AbstractQAtlas.peierls_phase`. Written out
# here rather than taken from the model so the two sides stay independent.
function _abstract_bond(m::RiceMele1D, i::Int, j::Int)
    t = isodd(i) ? m.v : m.w
    phase = peierls_phase(VectorPotential(m.A), bond_displacement(m, i, j))
    H = OpSum()
    H += (-t * cis(-phase), "Cdag", i, "C", j)
    H += (-t * cis(phase), "Cdag", j, "C", i)
    iszero(m.V) || (H += (m.V, "N", i, "N", j))
    return H
end

function _abstract_bond(m::RiceMeleHubbard1D, i::Int, j::Int)
    t = isodd(i) ? m.v : m.w
    phase = peierls_phase(VectorPotential(m.A), bond_displacement(m, i, j))
    H = OpSum()
    for (dag, ann) in (("Cdagup", "Cup"), ("Cdagdn", "Cdn"))
        H += (-t * cis(-phase), dag, i, ann, j)
        H += (-t * cis(phase), dag, j, ann, i)
    end
    return H
end

@testset "bond_displacement is the length unit, and it is one site spacing" begin
    for m in (RiceMele1D(; A=0.35), RiceMeleHubbard1D(; A=0.35))
        @test bond_displacement(m, 1, 2) == (1.0,)
        # Signed, and not restricted to pairs the model actually couples.
        @test bond_displacement(m, 2, 1) == (-1.0,)
        @test bond_displacement(m, 1, 3) == (2.0,)
    end
end

@testset "the models' Peierls phase is AbstractQAtlas.peierls_phase" begin
    # Nothing else in this suite can see a wrong length unit. Rescaling it rescales `A`, so it
    # cancels out of every static quantity and out of the linear and second-order responses; it
    # separates only at third order in the drive. So it is pinned here, directly.
    #
    # The NON-ADJACENT pairs carry the check. On `(1, 2)` the displacement is 1, so a model that
    # ignored `bond_displacement` and wrote `cis(-m.A)` would agree anyway — see the control
    # below, which asserts exactly that blindness for `(1, 3)`.
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9, A=0.35), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0, A=0.35), "Electron"),
    )
        sites = siteinds(st, 4)
        for (i, j) in ((1, 2), (2, 3), (1, 3), (2, 4))
            got = prod(MPO(bond_coupling_term(m, i, j), sites))
            want = prod(MPO(_abstract_bond(m, i, j), sites))
            @test norm(got - want) < 1.0e-12 * max(norm(want), 1.0)
        end
    end
end

@testset "the comparison above can fail: a phase that ignores the displacement" begin
    # `A` alone, i.e. a displacement forced to 1. Agrees on nearest neighbours and disagrees
    # everywhere else, which is what makes the non-adjacent pairs above load-bearing.
    m = RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9, A=0.35)
    sites = siteinds("Fermion", 4)
    flat(i, j) =
        let t = isodd(i) ? m.v : m.w
            H = OpSum()
            H += (-t * cis(-m.A), "Cdag", i, "C", j)
            H += (-t * cis(m.A), "Cdag", j, "C", i)
            iszero(m.V) || (H += (m.V, "N", i, "N", j))
            H
        end
    got13 = prod(MPO(bond_coupling_term(m, 1, 3), sites))
    @test norm(got13 - prod(MPO(flat(1, 3), sites))) > 0.1
    # ...and it really is only the displacement that differs: on a nearest-neighbour bond the
    # two spellings coincide, so this control cannot be passing for some unrelated reason.
    got12 = prod(MPO(bond_coupling_term(m, 1, 2), sites))
    @test norm(got12 - prod(MPO(flat(1, 2), sites))) < 1.0e-12
end

@testset "a zero field builds a real operator" begin
    # The `iszero(m.A)` shortcut in `_hop_amplitudes` exists to keep the undriven model real;
    # routing the phase through `bond_displacement` must not have cost that.
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0), "Electron"),
    )
        @test eltype(prod(MPO(bond_coupling_term(m, 1, 3), siteinds(st, 4)))) <: Real
    end
end
