using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LatticeCore
using LatticeCore: num_sites, position, bonds
using Lattice2D
using QuasiCrystal
using QuasiCrystal: build_nearest_neighbor_bonds!
using Random
using Test

@testset "TightBinding1D on Lattice2D.honeycomb via LatticeModel" begin
    lat = Lattice2D.honeycomb(3, 3; boundary=OpenAxis())
    sub = TightBinding1D(; t=1.0, μ=0.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    @test base isa LatticeModel
    @test ITensorModels.site_type(base) == SiteType("Fermion")

    H = build_opsum(base, nothing)
    nbonds = length(collect(bonds(lat)))
    @test length(collect(ITensors.terms(H))) > 0
    @test length(collect(ITensors.terms(H))) >= 2 * nbonds
end

@testset "TightBinding1D on QuasiCrystal.fibonacci via LatticeModel" begin
    lat = fibonacci(4)
    build_nearest_neighbor_bonds!(lat; cutoff=2.0)
    sub = TightBinding1D(; t=1.0, μ=0.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    H = build_opsum(base, nothing)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "TightBinding1D + spherical_ssd on honeycomb" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    sub = TightBinding1D(; t=1.0, μ=0.5)
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

@testset "TightBinding1D 2D DMRG smoke" begin
    lat = Lattice2D.honeycomb(2, 2; boundary=OpenAxis())
    sub = TightBinding1D(; t=1.0, μ=0.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    N = num_sites(lat)
    sites = siteinds("Fermion", N)
    H = MPO(build_opsum(base, sites), sites)
    psi0 = random_mps(MersenneTwister(0xb7), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-10)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
