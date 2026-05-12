using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "S1AnisotropicD1D: construction" begin
    m = S1AnisotropicD1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.D == 0.0
    @test site_type(m) == SiteType("S=1")
end

@testset "S1AnisotropicD1D: bond_coupling_term has 3 (XX/YY/ZZ)" begin
    m = S1AnisotropicD1D(; J=1.5, D=0.7)
    H = bond_coupling_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 3
end

@testset "S1AnisotropicD1D: onsite_term has 1 (D Sz Sz)" begin
    m = S1AnisotropicD1D(; J=1.0, D=0.5)
    H = onsite_term(m, 3)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 1
    @test ITensors.coefficient(terms[1]) ≈ 0.5
end

@testset "S1AnisotropicD1D: onsite_observable_op" begin
    m = S1AnisotropicD1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "S1AnisotropicD1D: build_opsum + ModulatedModel smoke" begin
    L = 6
    m = S1AnisotropicD1D(; J=1.0, D=0.5)
    m_ssd = modulated(m; L=L, modulation=SSD())
    sites = siteinds("S=1", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "S1AnisotropicD1D: large-D limit DMRG smoke" begin
    # Large D ≫ J: GS approaches the trivial product |Sz=0⟩^N with E ≈ 0.
    N = 6
    m = S1AnisotropicD1D(; J=0.1, D=10.0)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xae), sites; linkdims=12)
    sweeps = Sweeps(20)
    maxdim!(
        sweeps,
        20,
        50,
        100,
        150,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
        200,
    )
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E > -1.0
end
