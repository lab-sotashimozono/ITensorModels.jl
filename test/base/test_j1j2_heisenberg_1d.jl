using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "J1J2Heisenberg1D: construction" begin
    m = J1J2Heisenberg1D()
    @test m isa AbstractLatticeModel
    @test m.J1 == 1.0
    @test m.J2 == 0.5
    @test site_type(m) == SiteType("S=1/2")
end

@testset "J1J2Heisenberg1D: bond_term emits only NN piece" begin
    m = J1J2Heisenberg1D(; J1=1.0, J2=0.3)
    H = bond_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 3
    @test all(t -> ITensors.coefficient(t) ≈ 1.0, terms)
end

@testset "J1J2Heisenberg1D: local_ham_terms emits NN + NNN" begin
    m = J1J2Heisenberg1D(; J1=1.0, J2=0.5)
    N = 6
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == (N - 1) + (N - 2)
end

@testset "J1J2Heisenberg1D: build_opsum on S=1/2 sites" begin
    N = 6
    m = J1J2Heisenberg1D(; J1=1.0, J2=0.5)
    sites = siteinds("S=1/2", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == 3 * ((N - 1) + (N - 2))
end

@testset "J1J2Heisenberg1D: Majumdar-Ghosh DMRG smoke" begin
    N = 8
    m = J1J2Heisenberg1D(; J1=1.0, J2=0.5)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xcb), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0.0
end
