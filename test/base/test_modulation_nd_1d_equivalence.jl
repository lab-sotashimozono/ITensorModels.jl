# Regression test: ND modulation on a 1D LineLattice must reproduce the
# Hamiltonian of the existing 1D `ModulatedModel(TFIM(), L, SSD())`.
#
# The two code paths use different OpSum layouts (1D half-distributes
# on-site terms across two adjacent bonds and adds boundary patches;
# ND emits per-bond and per-site terms separately), so the OpSums are
# not byte-identical. Instead we build MPOs from both and verify they
# agree on a battery of random MPS states via `inner(psi', H, psi)`.

using ITensorModels
using LatticeCore
using LatticeCore: num_sites
using ITensors
using ITensorMPS: random_mps, MPO
using Random
using Test

@testset "ND on LineLattice ≡ 1D ModulatedModel(SSD) — site/bond weights" begin
    L = 10
    # The ND envelope on LineLattice (sites at integer 1..L,
    # BoundingBoxCenter at (L+1)/2) with SinSquareProfile(L/2) is
    # designed to match the 1D SSD weights exactly.
    lat = LineLattice(L, OpenAxis())
    env = RadialEnvelope(
        BoundingBoxCenter(), AxialDistance(1), SinSquareProfile(L / 2)
    )

    # site_weight on ND ≡ 1D site_weight(SSD, k, L).
    for k in 1:L
        nd_w = ITensorModels.site_weight(env, lat, k)
        oned_w = ITensorModels.site_weight(SSD(), k, L)
        @test nd_w ≈ oned_w
    end

    # bond_weight on ND ≡ 1D bond_weight(SSD, i, L) (bond between sites i and i+1).
    for i in 1:(L - 1)
        nd_w = ITensorModels.bond_weight(env, lat, i, i + 1)
        oned_w = ITensorModels.bond_weight(SSD(), i, L)
        @test nd_w ≈ oned_w
    end
end

@testset "ND on LineLattice ≡ 1D ModulatedModel(SSD) — MPO matrix elements" begin
    L = 6
    J = 1.0
    h = 0.7

    sites = siteinds("S=1/2", L)

    # 1D ModulatedModel path.
    base_1d = TFIM(; J=J, h=h)
    mod_1d = modulated(base_1d; L=L, modulation=SSD())
    H_1d_opsum = build_opsum(mod_1d, sites; phys_sites=1:L, boundary=:full)
    MPO_1d = MPO(H_1d_opsum, sites)

    # ND path: LineLattice + LatticeModel(TFIM) + RadialEnvelope.
    lat = LineLattice(L, OpenAxis())
    base_nd = LatticeModel(;
        lattice=lat,
        bond_models=Dict(:nearest => TFIM(; J=J, h=h)),
    )
    env = RadialEnvelope(
        BoundingBoxCenter(), AxialDistance(1), SinSquareProfile(L / 2)
    )
    mod_nd = modulated_lattice(base_nd; envelope=env)
    H_nd_opsum = build_opsum(mod_nd, sites)
    MPO_nd = MPO(H_nd_opsum, sites)

    # Compare ⟨ψ|H|ψ⟩ on a battery of random MPS states.
    rng = MersenneTwister(0x4d44)
    for trial in 1:5
        psi = random_mps(rng, sites; linkdims=4)
        e_1d = real(inner(psi', MPO_1d, psi))
        e_nd = real(inner(psi', MPO_nd, psi))
        @test e_1d ≈ e_nd atol = 1e-9
    end
end

@testset "ND on LineLattice with large-R envelope ≡ bare TFIM" begin
    # When R is much larger than L, 1 - sin²(π d / (2R)) ≈ 1 at every
    # site, so the ND envelope reduces to the unmodulated TFIM up to
    # the leading 1e-3 residue from finite R. Compare to the 1D
    # ModulatedModel(Uniform()) which trivially has weight 1 everywhere.
    L = 6
    J = 1.0
    h = 0.5

    sites = siteinds("S=1/2", L)

    lat = LineLattice(L, OpenAxis())
    base_nd = LatticeModel(;
        lattice=lat,
        bond_models=Dict(:nearest => TFIM(; J=J, h=h)),
    )
    env = RadialEnvelope(
        BoundingBoxCenter(), AxialDistance(1), SinSquareProfile(1000 * L)
    )
    mod_nd = modulated_lattice(base_nd; envelope=env)
    MPO_nd_huge = MPO(build_opsum(mod_nd, sites), sites)

    mod_uniform = modulated(TFIM(; J=J, h=h); L=L, modulation=Uniform())
    MPO_uniform = MPO(build_opsum(mod_uniform, sites; phys_sites=1:L, boundary=:full), sites)

    rng = MersenneTwister(0x9c1f)
    for trial in 1:3
        psi = random_mps(rng, sites; linkdims=4)
        e_huge = real(inner(psi', MPO_nd_huge, psi))
        e_unif = real(inner(psi', MPO_uniform, psi))
        @test e_huge ≈ e_unif atol = 1e-4
    end
end
