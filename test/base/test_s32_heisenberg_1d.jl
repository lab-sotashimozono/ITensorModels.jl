using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "S32Heisenberg1D: construction" begin
    m = S32Heisenberg1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test site_type(m) == SiteType("S=3/2")
end

@testset "S32Heisenberg1D: bond_term has 3 equal-J terms" begin
    m = S32Heisenberg1D(; J=0.5)
    H = bond_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 3
    coefs = [ITensors.coefficient(t) for t in terms]
    @test all(c ≈ 0.5 for c in coefs)
end

@testset "S32Heisenberg1D: onsite and boundary empty" begin
    m = S32Heisenberg1D()
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
    @test length(collect(ITensors.terms(boundary_patch(m, 1)))) == 0
end

@testset "S32Heisenberg1D: onsite_observable_op" begin
    m = S32Heisenberg1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "S32Heisenberg1D: build_opsum non-empty" begin
    N = 4
    m = S32Heisenberg1D(; J=1.0)
    sites = siteinds("S=3/2", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 3 * (N - 1)
end

@testset "S32Heisenberg1D: AFM DMRG smoke" begin
    N = 6
    m = S32Heisenberg1D(; J=1.0)
    sites = siteinds("S=3/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xc1), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "S32Heisenberg1D: FM DMRG smoke" begin
    N = 6
    m = S32Heisenberg1D(; J=-1.0)
    sites = siteinds("S=3/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xc2), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
