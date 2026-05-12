using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "LongRangeIsing1D: construction" begin
    m = LongRangeIsing1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.α == 3.0
    @test m.h == 0.0
    @test site_type(m) == SiteType("Qubit")
end

@testset "LongRangeIsing1D: pairwise / onsite bond_term empty" begin
    m = LongRangeIsing1D(; J=1.0, α=2.5, h=0.3)
    @test length(collect(ITensors.terms(bond_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(bond_coupling_term(m, 1, 2)))) == 0
end

@testset "LongRangeIsing1D: onsite_term has 1 (-h X)" begin
    m = LongRangeIsing1D(; J=1.0, α=3.0, h=0.7)
    H = onsite_term(m, 4)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 1
    @test ITensors.coefficient(terms[1]) ≈ -0.7
end

@testset "LongRangeIsing1D: local_ham_terms emits N(N-1)/2 ZZ + N X" begin
    N = 5
    m = LongRangeIsing1D(; J=1.0, α=3.0, h=0.5)
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == div(N * (N - 1), 2) + N
end

@testset "LongRangeIsing1D: local_ham_terms with h=0 drops X tail" begin
    N = 4
    m = LongRangeIsing1D(; J=1.0, α=3.0, h=0.0)
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == div(N * (N - 1), 2)
end

@testset "LongRangeIsing1D: onsite_observable_op" begin
    m = LongRangeIsing1D()
    @test onsite_observable_op(m, :x) == "X"
    @test onsite_observable_op(m, :y) == "Y"
    @test onsite_observable_op(m, :z) == "Z"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "LongRangeIsing1D: α large recovers near-NN TFIM (DMRG smoke)" begin
    # With very large α, only nearest-neighbor ZZ couplings are numerically
    # relevant; the GS is a ferromagnetic / paramagnetic competition.
    N = 6
    m = LongRangeIsing1D(; J=1.0, α=20.0, h=0.5)
    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xdf), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "LongRangeIsing1D: build_opsum on Qubit sites" begin
    N = 5
    m = LongRangeIsing1D(; J=1.0, α=2.0, h=0.3)
    sites = siteinds("Qubit", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end
