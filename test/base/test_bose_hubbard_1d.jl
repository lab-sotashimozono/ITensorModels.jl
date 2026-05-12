using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "BoseHubbard1D: construction" begin
    m = BoseHubbard1D()
    @test m isa AbstractLatticeModel
    @test m.t == 1.0
    @test m.U == 2.0
    @test m.μ == 0.0
    @test site_type(m) == SiteType("Boson")
end

@testset "BoseHubbard1D: bond_coupling_term has 2 hoppings" begin
    m = BoseHubbard1D(; t=1.3)
    H = bond_coupling_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 2
    @test all(t -> ITensors.coefficient(t) ≈ -1.3, terms)
end

@testset "BoseHubbard1D: onsite_term has 2 terms" begin
    m = BoseHubbard1D(; t=1.0, U=3.0, μ=0.5)
    H = onsite_term(m, 1)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 2
end

@testset "BoseHubbard1D: onsite_observable_op" begin
    m = BoseHubbard1D()
    @test onsite_observable_op(m, :n) == "N"
    @test onsite_observable_op(m, :a) == "A"
    @test onsite_observable_op(m, :adag) == "Adag"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "BoseHubbard1D: build_opsum on Boson sites" begin
    N = 6
    m = BoseHubbard1D(; t=1.0, U=2.0, μ=-1.0)
    sites = siteinds("Boson", N; dim=3)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "BoseHubbard1D: DMRG smoke" begin
    N = 6
    m = BoseHubbard1D(; t=1.0, U=8.0, μ=-4.0)
    sites = siteinds("Boson", N; dim=3)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xbe), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
