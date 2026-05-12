using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using Random
using Test

@testset "XYh1D: construction" begin
    m = XYh1D()
    @test m isa AbstractLatticeModel
    @test m.J == 1.0
    @test m.γ == 0.0
    @test m.h == 0.0
    @test site_type(m) == SiteType("S=1/2")
end

@testset "XYh1D: bond_coupling_term has 2 (XX + YY)" begin
    m = XYh1D(; J=1.0, γ=0.3, h=0.5)
    H = bond_coupling_term(m, 1, 2)
    @test length(collect(ITensors.terms(H))) == 2
end

@testset "XYh1D: bond_term coefficients (split-protocol)" begin
    # With h ≠ 0 the bond_term must carry the two-body coupling at full
    # value plus a half-edge -h/2 Sz on each endpoint, so that summing
    # over interior bonds gives the right -h Sz per site and the OBC
    # endpoints are completed by boundary_patch.
    J, γ, h = 1.0, 0.3, 0.5
    m = XYh1D(; J=J, γ=γ, h=h)
    H = bond_term(m, 1, 2)
    coefs = Float64[]
    for t in ITensors.terms(H)
        push!(coefs, ITensors.coefficient(t))
    end
    # Expected (order-independent): -J(1+γ), -J(1-γ), -h/2, -h/2
    expected = sort([-J * (1 + γ), -J * (1 - γ), -h / 2, -h / 2])
    @test sort(coefs) ≈ expected
end

@testset "XYh1D: boundary_patch is -h/2 Sz" begin
    m = XYh1D(; J=1.0, γ=0.0, h=0.4)
    H = boundary_patch(m, 1)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 1
    @test ITensors.coefficient(terms[1]) ≈ -0.2
end

@testset "XYh1D: onsite_term -h Sz" begin
    m = XYh1D(; J=1.0, γ=0.0, h=0.7)
    H = onsite_term(m, 3)
    terms = collect(ITensors.terms(H))
    @test length(terms) == 1
    @test ITensors.coefficient(terms[1]) ≈ -0.7
end

@testset "XYh1D: onsite_observable_op" begin
    m = XYh1D()
    @test onsite_observable_op(m, :sx) == "Sx"
    @test onsite_observable_op(m, :sy) == "Sy"
    @test onsite_observable_op(m, :sz) == "Sz"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "XYh1D: build_opsum + ModulatedModel smoke" begin
    L = 6
    m = XYh1D(; J=1.0, γ=0.5, h=0.3)
    m_ssd = modulated(m; L=L, modulation=SSD())
    sites = siteinds("S=1/2", L)
    H = build_opsum(m_ssd, sites; phys_sites=1:L, boundary=:full)
    @test length(collect(ITensors.terms(H))) > 0
end

@testset "XYh1D: γ=0 XX limit DMRG smoke" begin
    N = 6
    m = XYh1D(; J=1.0, γ=0.0, h=0.0)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0x7a), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end

@testset "XYh1D: γ=1 TFIM-like DMRG smoke" begin
    N = 6
    m = XYh1D(; J=1.0, γ=1.0, h=0.5)
    sites = siteinds("S=1/2", N)
    H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0x7b), sites; linkdims=12)
    sweeps = Sweeps(15)
    maxdim!(sweeps, 20, 50, 100, 150, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    cutoff!(sweeps, 1e-12)
    E, _ = dmrg(H, psi0, sweeps; outputlevel=0)
    @test isfinite(E)
    @test E < 0
end
