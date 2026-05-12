using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "S1XXZ1D: construction" begin
    m = S1XXZ1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.Delta == 1.0
    @test site_type(m) == SiteType("S=1")
end

@testset "S1XXZ1D: bond_term has 3 terms" begin
    m = S1XXZ1D(; J=1.0, Delta=0.5)
    H = bond_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 3
end

@testset "S1XXZ1D: Delta=1 all coefficients equal J" begin
    m = S1XXZ1D(; J=1.0, Delta=1.0)
    H = bond_term(m, 1, 2)
    coefs = sort([real(ITensors.coefficient(t)) for t in ITensors.terms(H)])
    @test coefs == [1.0, 1.0, 1.0]
end

@testset "S1XXZ1D: Delta scales Sz coupling" begin
    J, Delta = 1.0, 2.0
    m = S1XXZ1D(; J=J, Delta=Delta)
    H = bond_term(m, 1, 2)
    coefs = sort([real(ITensors.coefficient(t)) for t in ITensors.terms(H)])
    @test coefs ≈ sort([J, J, J * Delta])
end

@testset "S1XXZ1D: onsite and boundary empty" begin
    m = S1XXZ1D()
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
    @test length(collect(ITensors.terms(boundary_patch(m, 1)))) == 0
end

@testset "S1XXZ1D: onsite_observable_op" begin
    m = S1XXZ1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "S1XXZ1D: Haldane DMRG smoke" begin
    N = 6
    m = S1XXZ1D(; J=1.0, Delta=1.0)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xa1), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "S1XXZ1D: large-Delta Neel DMRG smoke" begin
    N = 6
    m = S1XXZ1D(; J=1.0, Delta=5.0)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xa2), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
