using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "TightBindingV1D: construction" begin
    m = TightBindingV1D()
    @test m isa AbstractLatticeModel
    @test m.t == 1.0
    @test m.V == 1.0
    @test m.μ == 0.0
    @test site_type(m) == SiteType("Fermion")
end

@testset "TightBindingV1D: bond_coupling_term has hopping + V" begin
    m = TightBindingV1D(; t=1.5, V=2.0, μ=0.5)
    H = bond_coupling_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 3
end

@testset "TightBindingV1D: onsite_term carries μ only" begin
    m = TightBindingV1D(; t=1.0, V=2.0, μ=0.5)
    H = onsite_term(m, 1)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 1
end

@testset "TightBindingV1D: onsite_observable_op" begin
    m = TightBindingV1D()
    @test onsite_observable_op(m, :n) == "N"
    @test onsite_observable_op(m, :c) == "C"
    @test onsite_observable_op(m, :cdag) == "Cdag"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "TightBindingV1D: build_opsum + ModulatedModel smoke" begin
    N = 6
    m = TightBindingV1D(; t=1.0, V=1.0, μ=-0.5)
    sites = siteinds("Fermion", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0

    m_ssd = modulated(m; L=N, modulation=SSD())
    H_ssd = build_opsum(m_ssd, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H_ssd))) > 0
end

@testset "TightBindingV1D: DMRG smoke" begin
    N = 8
    m = TightBindingV1D(; t=1.0, V=1.0, μ=0.0)
    sites = siteinds("Fermion", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0x7e), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
