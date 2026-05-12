using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "TightBinding1D: construction" begin
    m = TightBinding1D()
    @test m isa AbstractLatticeModel
    @test m.t == 1.0
    @test m.μ == 0.0
    @test site_type(m) == SiteType("Fermion")

    m2 = TightBinding1D(; t=2.5, μ=-0.5)
    @test m2.t == 2.5
    @test m2.μ == -0.5
end

@testset "TightBinding1D: bond_coupling_term has hopping only" begin
    m = TightBinding1D(; t=1.5, μ=0.7)
    H_coup = bond_coupling_term(m, 1, 2)
    terms = collect(ITensors.terms(H_coup))
    @test length(terms) == 2
    @test all(t -> ITensors.coefficient(t) ≈ -1.5, terms)
end

@testset "TightBinding1D: onsite_term carries full μ" begin
    m = TightBinding1D(; t=1.0, μ=0.7)
    H_on = onsite_term(m, 3)
    terms = collect(ITensors.terms(H_on))
    @test length(terms) == 1
    @test ITensors.coefficient(terms[1]) ≈ 0.7
end

@testset "TightBinding1D: onsite_observable_op" begin
    m = TightBinding1D()
    @test onsite_observable_op(m, :n) == "N"
    @test onsite_observable_op(m, :c) == "C"
    @test onsite_observable_op(m, :cdag) == "Cdag"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "TightBinding1D: build_opsum on Fermion sites" begin
    N = 6
    m = TightBinding1D(; t=1.0, μ=0.0)
    sites = siteinds("Fermion", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "TightBinding1D: ModulatedModel wraps without error" begin
    L = 6
    m_ssd = modulated(TightBinding1D(; t=1.0, μ=0.5); L=L, modulation=SSD())
    sites = siteinds("Fermion", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "TightBinding1D: DMRG GS energy matches analytic half-filling" begin
    # Open-chain spectrum: ε_k = -2t cos(kπ/(N+1)), k ∈ 1..N.
    # At μ = 0, half-filling fills the negative-ε states.
    N = 8
    t = 1.0
    m = TightBinding1D(; t=t, μ=0.0)
    sites = siteinds("Fermion", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

    eps = [-2 * t * cos(k * π / (N + 1)) for k in 1:N]
    E_exact = sum(filter(e -> e < 0, eps))

    psi0 = random_mps(MersenneTwister(0xfe), sites; linkdims=16)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 40, 80, 100, 120, 120, 120, 120, 120,
        120, 120, 120, 120, 120, 120, 120, 120, 120, 120)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test E_dmrg ≈ E_exact rtol = 1e-5
end
