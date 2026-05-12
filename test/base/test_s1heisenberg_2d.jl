using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LatticeCore
using LatticeCore: num_sites, position, positions, bonds
using Lattice2D
using Random
using Test

@testset "S1Heisenberg1D on Lattice2D.honeycomb via LatticeModel" begin
    lat = Lattice2D.honeycomb(3, 3; boundary=OpenAxis())
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    @test base isa LatticeModel
    @test site_type(base) == SiteType("S=1")

    H = build_opsum(base, nothing)
    nbonds = length(collect(bonds(lat)))
    # 3 (XX + YY + ZZ) per bond, no on-site for S=1 Heisenberg.
    @test length(collect(ITensors.terms(H))) == 3 * nbonds
end

@testset "S1Heisenberg1D on Lattice2D.kagome via LatticeModel" begin
    lat = Lattice2D.kagome(2, 2; boundary=OpenAxis())
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    H = build_opsum(base, nothing)
    nbonds = length(collect(bonds(lat)))
    @test length(collect(ITensors.terms(H))) == 3 * nbonds
    @test nbonds > 0
end

@testset "S1Heisenberg1D + spherical_ssd ND modulation on honeycomb" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    env = spherical_ssd(lat; radius=:circumscribed)
    mod = modulated_lattice(base; envelope=env)
    H = build_opsum(mod, nothing)
    @test length(collect(ITensors.terms(H))) > 0

    bs = collect(bonds(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    midpoints = [(position(lat, b.i) + position(lat, b.j)) / 2 for b in bs]
    distances = [sqrt(sum((mid .- rc) .^ 2)) for mid in midpoints]
    k_central = argmin(distances)
    w_corner = ITensorModels.bond_weight(env, lat, bs[1].i, bs[1].j)
    w_centre = ITensorModels.bond_weight(env, lat, bs[k_central].i, bs[k_central].j)
    @test w_corner < w_centre
end

@testset "S1Heisenberg1D 2D DMRG smoke (very small)" begin
    # Tiny honeycomb so DMRG converges quickly. We only check the
    # pipeline runs; not a benchmark.
    lat = Lattice2D.honeycomb(2, 2; boundary=OpenAxis())
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    N = num_sites(lat)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(base, sites), sites)
    psi0 = random_mps(MersenneTwister(0xa5), sites; linkdims=12)
    sweeps = Sweeps(12)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-10)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0.0
end
