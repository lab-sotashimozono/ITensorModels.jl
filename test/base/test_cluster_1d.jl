using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "Cluster1D: construction" begin
    m = Cluster1D()
    @test m isa AbstractLatticeModel
    @test m.h == 1.0
    @test site_type(m) == SiteType("Qubit")
end

@testset "Cluster1D: pairwise / onsite all empty (3-site)" begin
    m = Cluster1D()
    @test length(collect(ITensors.terms(bond_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(bond_coupling_term(m, 1, 2)))) == 0
    @test length(collect(ITensors.terms(onsite_term(m, 1)))) == 0
end

@testset "Cluster1D: local_ham_terms emits N-2 cluster terms" begin
    m = Cluster1D(; h=1.5)
    N = 5
    terms = local_ham_terms(m, 1:N; boundary=:full)
    @test length(terms) == N - 2
    for t in terms
        @test length(collect(ITensors.terms(t))) == 1
    end
end

@testset "Cluster1D: build_opsum on Qubit sites" begin
    N = 6
    m = Cluster1D(; h=1.0)
    sites = siteinds("Qubit", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) == N - 2
end

@testset "Cluster1D: onsite_observable_op" begin
    m = Cluster1D()
    @test onsite_observable_op(m, :x) == "X"
    @test onsite_observable_op(m, :y) == "Y"
    @test onsite_observable_op(m, :z) == "Z"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "Cluster1D: DMRG GS energy = -(N-2)*h" begin
    # Cluster state is the exact +1 eigenstate of every ZXZ stabilizer,
    # so H|cluster⟩ = -(N-2) h |cluster⟩.
    N = 6
    h = 1.0
    m = Cluster1D(; h=h)
    sites = siteinds("Qubit", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xc1), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test E ≈ -(N - 2) * h rtol = 1e-6
end
