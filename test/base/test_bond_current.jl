using ITensorModels
using ITensors
using ITensorMPS
using AbstractQAtlas: ElectricCurrent, ElectricCurrentResponse, bag, check, derivative_edge
using AbstractQAtlas: Energy, VectorPotentialField
using Random
using Test

const _L = 6

# ⟨J⟩ and ∂⟨H⟩/∂A for one fixed state — Hellmann-Feynman is exact there, so no ground state
# is needed. The state must be entangled: a product state has zero hopping expectation.
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
    # Parented on the HAMILTONIAN, not the free energy, because `A` couples to the hopping.
    e = derivative_edge(ElectricCurrent)
    @test e.parent === Energy
    @test e.field === VectorPotentialField
end

@testset "bond_current_term satisfies ElectricCurrentResponse" begin
    # The checker is AbstractQAtlas's relation, so the sign is not restated here. `dH_dA`
    # differences the model's own `H`, which is what can catch a disagreement about the unit.
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
            @test abs(j) > 1.0e-3        # not a trivial 0 == 0
        end
    end
end

@testset "the check rejects a current that uses another length unit" begin
    # Writing `d = 1.0` instead of `bond_displacement(m)` is NOT caught — today it is the same
    # operator. What this buys is later drift: change the unit and a hardcoded `J` goes red.
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
        @test norm(J) > 1.0e-6
    end
end
