using ITensorModels
using ITensors
using ITensorMPS
using AbstractQAtlas: ElectricCurrent, ElectricCurrentResponse, bag, check, derivative_edge
using AbstractQAtlas: Energy, VectorPotentialField
using Random
using Test

const _L = 6

# ⟨J⟩ and ∂⟨H⟩/∂A for one fixed state. Hellmann-Feynman is exact for a FIXED state, so this
# needs no ground state — and a random complex MPS is used deliberately: a product state in
# the occupation basis has zero hopping expectation, which would make every comparison below
# 0 == 0.
function _j_and_dH(m, sites, psi; h=1.0e-5)
    L = length(sites)
    Jsum = sum(i -> bond_current_term(m, i, i + 1), 1:(L - 1))
    j = real(inner(psi', MPO(Jsum, sites), psi))
    E(a) = real(
        inner(
            psi',
            MPO(build_opsum(_reset_A(m, a), sites; phys_sites=1:L, boundary=:full), sites),
            psi,
        ),
    )
    return j, (E(m.A + h) - E(m.A - h)) / 2h
end
_reset_A(m::RiceMele1D, a) = RiceMele1D(; v=m.v, w=m.w, Δ=m.Δ, V=m.V, A=a, site=m.site)
function _reset_A(m::RiceMeleHubbard1D, a)
    return RiceMeleHubbard1D(; v=m.v, w=m.w, Δ=m.Δ, U=m.U, A=a, site=m.site)
end

@testset "the relation implemented here is the one AbstractQAtlas declares" begin
    # `j = -∂H/∂A`, parented on the HAMILTONIAN rather than the free energy because `A` couples
    # to the hopping. Naming it here means a change to that edge upstream lands as a red test
    # rather than as a silently different sign.
    e = derivative_edge(ElectricCurrent)
    @test e.parent === Energy
    @test e.field === VectorPotentialField
end

@testset "bond_current_term satisfies ElectricCurrentResponse" begin
    # The check is AbstractQAtlas's own relation — `j - (-dH_dA)` — so the sign is not
    # restated here. `dH_dA` comes from differentiating the model's own Hamiltonian, which is
    # the one thing that can catch the current disagreeing with `H` about the length unit.
    Random.seed!(11)
    for (m0, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0), "Electron"),
    )
        sites = siteinds(st, _L)
        psi = random_mps(ComplexF64, sites; linkdims=8)
        for A in (0.0, 0.35, 0.8)
            m = _reset_A(m0, A)
            j, dH_dA = _j_and_dH(m, sites, psi)
            @test check(
                ElectricCurrentResponse(),
                bag(ElectricCurrent => j);
                dH_dA=dH_dA,
                atol=1.0e-6,
            )
            @test abs(j) > 1.0e-3        # not the trivial 0 == 0 agreement
        end
    end
end

@testset "the check rejects a current that uses another length unit" begin
    # `J` scales linearly with the unit, so a current built with a different one fails the
    # relation while every static and A = 0 check stays green. Measured, by mutating
    # `bond_current_term`: a wrong unit (`d = 0.5`) and a flipped sign are both caught.
    #
    # What is NOT caught is writing `d = 1.0` in place of `bond_displacement(m)` — today that
    # is the same operator, so no test can see it. The protection this buys is against LATER
    # drift: change the unit in one place and `H` moves while a hardcoded `J` does not, and
    # this relation is what turns red.
    Random.seed!(11)
    m = RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9, A=0.35)
    sites = siteinds("Fermion", _L)
    psi = random_mps(ComplexF64, sites; linkdims=8)
    j, dH_dA = _j_and_dH(m, sites, psi)
    @test check(
        ElectricCurrentResponse(), bag(ElectricCurrent => j); dH_dA=dH_dA, atol=1.0e-6
    )
    for scale in (0.5, 2.0)
        @test !check(
            ElectricCurrentResponse(),
            bag(ElectricCurrent => scale * j);
            dH_dA=dH_dA,
            atol=1.0e-6,
        )
    end
end

@testset "the current is Hermitian and carries the displacement linearly" begin
    for (m, st) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4, A=0.35), "Fermion"),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0, A=0.35), "Electron"),
    )
        sites = siteinds(st, 3)
        J = prod(MPO(bond_current_term(m, 1, 2), sites))
        @test norm(J - swapprime(dag(J), 0 => 1)) < 1.0e-12 * norm(J)
        # An observable, so it must not be the zero operator dressed up as one.
        @test norm(J) > 1.0e-6
    end
end
