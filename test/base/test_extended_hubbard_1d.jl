using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "ExtendedHubbard1D: construction" begin
    m = ExtendedHubbard1D()
    @test m isa AbstractLatticeModel
    @test m.t == 1.0
    @test m.U == 4.0
    @test m.V == 1.0
    @test m.μ == 0.0
    @test site_type(m) == SiteType("Electron")
end

@testset "ExtendedHubbard1D: bond_coupling_term has 5 (4 hop + 1 V)" begin
    m = ExtendedHubbard1D(; t=1.0, U=4.0, V=1.5)
    H = bond_coupling_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 5
end

@testset "ExtendedHubbard1D: onsite_term has U Nupdn + μ Ntot" begin
    m = ExtendedHubbard1D(; t=1.0, U=2.0, V=0.0, μ=0.3)
    H = onsite_term(m, 3)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 2
end

@testset "ExtendedHubbard1D: onsite_observable_op" begin
    m = ExtendedHubbard1D()
    @test onsite_observable_op(m, :nup) == "Nup"
    @test onsite_observable_op(m, :ndn) == "Ndn"
    @test onsite_observable_op(m, :ntot) == "Ntot"
    @test onsite_observable_op(m, :nupdn) == "Nupdn"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "ExtendedHubbard1D: build_opsum + ModulatedModel smoke" begin
    L = 6
    m = ExtendedHubbard1D(; t=1.0, U=4.0, V=1.0)
    m_ssd = modulated(m; L=L, modulation=SSD())
    sites = siteinds("Electron", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "ExtendedHubbard1D: V → 0 reduces to standard Hubbard (DMRG smoke)" begin
    N = 6
    m = ExtendedHubbard1D(; t=1.0, U=8.0, V=0.0, μ=0.0)
    sites = siteinds("Electron", N; conserve_qns=false)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xfa), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end

@testset "ExtendedHubbard1D: large V favors CDW (DMRG smoke)" begin
    N = 6
    m = ExtendedHubbard1D(; t=1.0, U=1.0, V=4.0)
    sites = siteinds("Electron", N; conserve_qns=false)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0xfb), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
end
