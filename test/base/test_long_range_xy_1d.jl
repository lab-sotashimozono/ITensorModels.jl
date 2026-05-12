using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "LongRangeXY1D: construction" begin
    m = LongRangeXY1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.α == 3.0
    @test site_type(m) == SiteType("S=1/2")
end

@testset "LongRangeXY1D: bond_term has 2 (NN XX/YY only)" begin
    m = LongRangeXY1D(; J=1.5)
    H = bond_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 2
end

@testset "LongRangeXY1D: local_ham_terms emits all N(N-1)/2 pairs" begin
    m = LongRangeXY1D(; J=1.0, α=3.0)
    N = 5
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == N * (N - 1) ÷ 2
end

@testset "LongRangeXY1D: build_opsum yields 2 * N(N-1)/2 terms" begin
    m = LongRangeXY1D(; J=1.0, α=20.0)
    N = 6
    sites = siteinds("S=1/2", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 2 * (N * (N - 1) ÷ 2)
end

@testset "LongRangeXY1D: onsite_observable_op" begin
    m = LongRangeXY1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "LongRangeXY1D: DMRG smoke" begin
    N = 8
    m = LongRangeXY1D(; J=1.0, α=3.0)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xab), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
