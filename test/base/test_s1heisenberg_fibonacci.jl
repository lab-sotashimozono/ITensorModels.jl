using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LatticeCore
using LatticeCore: num_sites, position, bonds
using QuasiCrystal
using QuasiCrystal: build_nearest_neighbor_bonds!
using Random
using Test

@testset "S1Heisenberg1D on QuasiCrystal.fibonacci via LatticeModel" begin
    lat = fibonacci(4)
    build_nearest_neighbor_bonds!(lat; cutoff=2.0)
    @test lat isa LatticeCore.AbstractLattice
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    @test base isa LatticeModel
    @test ITensorModels.site_type(base) == SiteType("S=1")

    H = build_opsum(base, nothing)
    nbonds = length(collect(bonds(lat)))
    @test nbonds > 0
    @test length(collect(ITensors.terms(H))) == 3 * nbonds
end

@testset "S1Heisenberg1D + spherical_ssd on fibonacci" begin
    lat = fibonacci(4)
    build_nearest_neighbor_bonds!(lat; cutoff=2.0)
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    env = spherical_ssd(lat; radius=:circumscribed)
    mod = modulated_lattice(base; envelope=env)
    H = build_opsum(mod, nothing)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "S1Heisenberg1D DMRG on Fibonacci smoke" begin
    lat = fibonacci(3)
    build_nearest_neighbor_bonds!(lat; cutoff=2.0)
    sub = S1Heisenberg1D(; J=1.0)
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => sub))
    N = num_sites(lat)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(base, sites), sites)
    psi0 = random_mps(MersenneTwister(0xfa), sites; linkdims=10)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 20, 50, 100, 100, 100, 100, 100, 100, 100, 100)
    cutoff!(sweeps, 1e-10)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0.0
end
