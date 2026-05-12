using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "AKLT1D: construction" begin
    m = AKLT1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test site_type(m) == SiteType("S=1")

    m2 = AKLT1D(; J=2.0)
    @test m2.J == 2.0
end

@testset "AKLT1D: bond_term has 12 terms (3 bilinear + 9 biquadratic)" begin
    m = AKLT1D(; J=1.5)
    H = bond_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 12
end

@testset "AKLT1D: bond_coupling_term ≡ bond_term (no onsite)" begin
    m = AKLT1D(; J=0.8)
    @test length(collect(ITensors.terms(bond_term(m, 3, 4)))) ==
        length(collect(ITensors.terms(bond_coupling_term(m, 3, 4))))
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "AKLT1D: build_opsum on S=1 sites" begin
    N = 6
    m = AKLT1D(; J=1.0)
    sites = siteinds("S=1", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 12 * (N - 1)
end

@testset "AKLT1D: ModulatedModel wraps without error" begin
    L = 6
    m_ssd = modulated(AKLT1D(; J=1.0); L=L, modulation=SSD())
    sites = siteinds("S=1", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "AKLT1D: DMRG smoke" begin
    N = 8
    m = AKLT1D(; J=1.0)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xa1), sites; linkdims=8)
    sweeps = Sweeps(20)
    maxdim!(
        sweeps,
        10,
        20,
        40,
        80,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    )
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0.0
end
