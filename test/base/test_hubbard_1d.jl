using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "Hubbard1D: construction" begin
    m = Hubbard1D()
    @test m isa AbstractLatticeModel
    @test m.t == 1.0
    @test m.U == 4.0
    @test m.μ == 0.0
    @test site_type(m) == SiteType("Electron")
end

@testset "Hubbard1D: bond_coupling_term has 4 hoppings" begin
    m = Hubbard1D(; t=2.0)
    H = bond_coupling_term(m, 1, 2)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 4
    @test all(t -> ITensors.coefficient(t) ≈ -2.0, terms)
end

@testset "Hubbard1D: onsite_term carries U + 2μ" begin
    m = Hubbard1D(; t=1.0, U=3.5, μ=0.7)
    H = onsite_term(m, 1)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 3
end

@testset "Hubbard1D: onsite_observable_op" begin
    m = Hubbard1D()
    @test onsite_observable_op(m, :nup) == "Nup"
    @test onsite_observable_op(m, :ndn) == "Ndn"
    @test onsite_observable_op(m, :n) == "Ntot"
    @test onsite_observable_op(m, :nupdn) == "Nupdn"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "Hubbard1D: build_opsum on Electron sites" begin
    N = 4
    m = Hubbard1D(; t=1.0, U=4.0, μ=-2.0)
    sites = siteinds("Electron", N)
    H = build_opsum(m, sites; phys_sites=1:N, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "Hubbard1D: ModulatedModel wraps without error" begin
    L = 4
    m_ssd = modulated(Hubbard1D(; t=1.0, U=4.0); L=L, modulation=SSD())
    sites = siteinds("Electron", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "Hubbard1D: DMRG smoke" begin
    N = 4
    t = 1.0
    U = 4.0
    m = Hubbard1D(; t=t, U=U, μ=(-U / 2))
    sites = siteinds("Electron", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0x1c), sites; linkdims=16)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
