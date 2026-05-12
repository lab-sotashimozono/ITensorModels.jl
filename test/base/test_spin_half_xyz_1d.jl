using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "SpinHalfXYZ1D: construction" begin
    m = SpinHalfXYZ1D()
    @test m isa AbstractLatticeModel
    @test m.Jx == 1.0
    @test m.Jy == 1.0
    @test m.Jz == 1.0
    @test site_type(m) == SiteType("S=1/2")
end

@testset "SpinHalfXYZ1D: bond_term has 3 terms" begin
    m = SpinHalfXYZ1D(; Jx=0.5, Jy=0.7, Jz=1.2)
    H = bond_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 3
end

@testset "SpinHalfXYZ1D: coefficients match Jx Jy Jz" begin
    Jx, Jy, Jz = 0.3, 0.7, 1.5
    m = SpinHalfXYZ1D(; Jx=Jx, Jy=Jy, Jz=Jz)
    H = bond_term(m, 1, 2)
    coefs = sort([real(ITensors.coefficient(t)) for t in ITensors.terms(H)])
    @test coefs ≈ sort([Jx, Jy, Jz])
end

@testset "SpinHalfXYZ1D: XXX limit equals Heisenberg" begin
    m = SpinHalfXYZ1D(; Jx=1.0, Jy=1.0, Jz=1.0)
    H = bond_term(m, 1, 2)
    coefs = sort([real(ITensors.coefficient(t)) for t in ITensors.terms(H)])
    @test coefs ≈ [1.0, 1.0, 1.0]
end

@testset "SpinHalfXYZ1D: onsite and boundary empty" begin
    m = SpinHalfXYZ1D()
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
    @test length(collect(ITensors.terms(boundary_patch(m, 1)))) == 0
end

@testset "SpinHalfXYZ1D: onsite_observable_op" begin
    m = SpinHalfXYZ1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "SpinHalfXYZ1D: XXZ limit DMRG smoke" begin
    N = 6
    m = SpinHalfXYZ1D(; Jx=1.0, Jy=1.0, Jz=0.5)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xb1), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "SpinHalfXYZ1D: XYZ DMRG smoke" begin
    N = 6
    m = SpinHalfXYZ1D(; Jx=0.5, Jy=0.7, Jz=1.0)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xb2), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
