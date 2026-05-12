using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "Compass1D: construction" begin
    m = Compass1D()
    @test m isa AbstractLatticeModel
    @test m.Jx == 1.0
    @test m.Jy == 1.0
    @test site_type(m) == SiteType("S=1/2")
end

@testset "Compass1D: pairwise / onsite all empty" begin
    m = Compass1D(; Jx=1.0, Jy=0.5)
    @test length(collect(ITensors.terms(bond_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(bond_coupling_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "Compass1D: local_ham_terms alternates XX / YY" begin
    N = 5
    m = Compass1D(; Jx=1.0, Jy=0.5)
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == N - 1
    for t in terms
        @test length(collect(ITensors.terms(t))) == 1
    end
end

@testset "Compass1D: build_opsum non-empty" begin
    N = 6
    m = Compass1D(; Jx=1.0, Jy=0.7)
    sites = siteinds("S=1/2", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == N - 1
end

@testset "Compass1D: onsite_observable_op" begin
    m = Compass1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "Compass1D: DMRG smoke (finite GS)" begin
    N = 6
    m = Compass1D(; Jx=-1.0, Jy=-1.0)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xc2), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
