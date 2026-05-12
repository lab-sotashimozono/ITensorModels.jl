using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "S1Heisenberg1D: construction" begin
    m = S1Heisenberg1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test site_type(m) == SiteType("S=1")

    m2 = S1Heisenberg1D(; J=2.5)
    @test m2.J == 2.5
end

@testset "S1Heisenberg1D: bond_term structure" begin
    m = S1Heisenberg1D(; J=1.5)
    H = bond_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 3
    coeffs = [ITensors.coefficient(t) for t in terms]
    @test all(c -> c ≈ 1.5, coeffs)
end

@testset "S1Heisenberg1D: bond_coupling_term ≡ bond_term (no onsite)" begin
    m = S1Heisenberg1D(; J=0.8)
    H_bond = bond_term(m, 3, 4)
    H_coup = bond_coupling_term(m, 3, 4)
    @test length(collect(ITensors.terms(H_bond))) == length(collect(ITensors.terms(H_coup)))
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "S1Heisenberg1D: onsite_observable_op" begin
    m = S1Heisenberg1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "S1Heisenberg1D: build_opsum on S=1 sites" begin
    N = 6
    m = S1Heisenberg1D(; J=1.0)
    sites = siteinds("S=1", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    # N-1 bonds * 3 (XX/YY/ZZ); no on-site or boundary terms.
    @test length(collect(ITensors.terms(H))) == 3 * (N - 1)
end

@testset "S1Heisenberg1D: ModulatedModel wraps without error" begin
    L = 6
    m_ssd = modulated(S1Heisenberg1D(; J=1.0); L=L, modulation=SSD())
    sites = siteinds("S=1", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
    @test length(collect(ITensors.terms(H))) == 3 * (L - 1)
end

@testset "S1Heisenberg1D: DMRG GS energy sanity" begin
    # Spin-1 Heisenberg OBC GS energy density ≈ -1.4 in the bulk;
    # for N = 8 expect E ≈ -10.5. We only check the OpSum -> MPO ->
    # DMRG pipeline produces an energy in the expected qualitative
    # range. QAtlas comparison lands in PR-S1B.
    N = 8
    m = S1Heisenberg1D(; J=1.0)
    sites = siteinds("S=1", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

    rng = MersenneTwister(0x51)
    psi0 = random_mps(rng, sites; linkdims=8)
    sweeps = Sweeps(10)
    maxdim!(sweeps, 10, 20, 40, 60, 80, 100, 100, 100, 100, 100)
    cutoff!(sweeps, 1e-10)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test -15.0 < E < -5.0
end
