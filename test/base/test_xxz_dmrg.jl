using ITensorModels: XXZ1D, Heisenberg1D, build_opsum
using ITensors, ITensorMPS
using QAtlas: QAtlas
using QAtlas: Energy, Infinite, OBC
using Random

# XXZ ground-state validation: compare ITensorMPS DMRG on the OpSum
# produced by build_opsum against QAtlas's analytic per-site values at
# the canonical Δ = 0 (XX / free fermion) and Δ = 1 (Heisenberg) points.
#
# Finite-N OBC GS energy differs from the thermodynamic-limit value by
# O(1/N) corrections, so we compare per-site energy with a loose rtol.

@testset "XXZ1D Δ=0 (XX / free fermion) DMRG vs QAtlas per-site" begin
    J = 1.0
    N = 24
    m = XXZ1D(; J=J, Δ=0.0)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

    # Half-filling initial state to pick the Sz=0 sector.
    state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    ψ0 = MPS(sites, state)

    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 40, 80, 120)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

    ε_qatlas = QAtlas.fetch(m, Energy(), Infinite())  # per site, TL
    @test E_dmrg / N ≈ ε_qatlas rtol = 0.03            # O(1/N) OBC correction
end

@testset "Heisenberg1D DMRG vs QAtlas (Hulthén per-site)" begin
    J = 1.0
    N = 24
    m = Heisenberg1D(; J=J)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

    state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    ψ0 = MPS(sites, state)

    sweeps = Sweeps(25)
    maxdim!(sweeps, 10, 20, 50, 100, 200)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

    # QAtlas removed `GroundStateEnergyDensity`, folding it into `Energy{:per_site}`
    # fetched without a `beta` — the alias name survives in its canonicaliser, so the
    # import merely stopped binding and the failure was `UndefVarError ... in Main` at
    # first USE rather than at the `using`. The replacement is the same call minus the
    # BC argument that QAtlas still does not want here:
    # `fetch(::Heisenberg1D, ::Energy{:per_site}; J) = J * (1//4 - log(2))`.
    ε_qatlas = QAtlas.fetch(QAtlas.Heisenberg1D(), Energy{:per_site}(); J=m.J)
    @test E_dmrg / N ≈ ε_qatlas rtol = 0.05
    @test ε_qatlas ≈ 0.25 - log(2) rtol = 1e-12
end
