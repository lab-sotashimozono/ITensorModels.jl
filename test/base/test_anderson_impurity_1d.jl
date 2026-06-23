using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LinearAlgebra: norm
using Test

@testset "AndersonImpurity1D: construction + semielliptic bath" begin
    m = semielliptic_anderson(; Nb=5, D=1.0, U=2.0)
    @test m isa AbstractLatticeModel
    @test site_type(m) == SiteType("Electron")
    @test m.εd == -1.0                       # particle-hole symmetric default
    @test m.U == 2.0
    @test length(m.thop) == 5 && length(m.eon) == 5
    @test all(m.eon .== 0.0)                 # semielliptic -> uniform chain
    @test m.thop[1] == 0.5 && all(m.thop[2:end] .== 0.5)   # V = D/2, bath hopping D/2

    # length mismatch is rejected
    @test_throws ErrorException AndersonImpurity1D(;
        εd=0.0, U=1.0, eon=[0.0], thop=[1.0, 1.0]
    )
end

@testset "AndersonImpurity1D: star_to_chain tridiagonalization" begin
    εb = [-0.6, -0.1, 0.3, 0.7]              # asymmetric so the first moment ≠ 0
    vb = [0.3, 0.5, 0.4, 0.2]
    eon, thop = star_to_chain(εb, vb)
    @test length(eon) == 4 && length(thop) == 4
    @test thop[1] ≈ norm(vb)                 # impurity↔bath hybridization = ‖vb‖
    @test isapprox(eon[1], sum(@. vb^2 * εb) / sum(abs2, vb); atol=1e-10)   # first Lanczos diagonal
    @test all(>(0), thop)                    # positive hoppings from Lanczos
end

@testset "AndersonImpurity1D: split protocol (coupling vs onsite)" begin
    m = AndersonImpurity1D(; εd=-1.0, U=2.0, eon=[0.3, -0.3], thop=[0.7, 0.5])
    H_coup = bond_coupling_term(m, 1, 2)
    @test length(collect(ITensors.terms(H_coup))) == 4          # 4 hoppings (2 spins × h.c.)
    @test all(t -> ITensors.coefficient(t) ≈ -0.7, collect(ITensors.terms(H_coup)))

    # impurity (site 1): full εd on each spin + U on double occupancy
    on1 = onsite_term(m, 1)
    coeffs1 = sort(real.(ITensors.coefficient.(collect(ITensors.terms(on1)))))
    @test coeffs1 ≈ sort([-1.0, -1.0, 2.0])
    # bath (site 2): on-site energy only, no U
    on2 = onsite_term(m, 2)
    coeffs2 = sort(real.(ITensors.coefficient.(collect(ITensors.terms(on2)))))
    @test coeffs2 ≈ sort([0.3, 0.3])
end

@testset "AndersonImpurity1D: onsite_observable_op" begin
    m = semielliptic_anderson(; Nb=3)
    @test onsite_observable_op(m, :n) == "Ntot"
    @test onsite_observable_op(m, :nupdn) == "Nupdn"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end

@testset "AndersonImpurity1D: U=0 chain = free fermions (analytic GS)" begin
    # U=0, εd=0, V=D/2 -> a uniform N-site tight-binding chain (hopping D/2),
    # two decoupled spins. ε_k = -D cos(kπ/(N+1)); half-filling fills ε<0.
    Nb = 7;
    D = 1.0;
    N = Nb + 1
    m = semielliptic_anderson(; Nb, D, U=0.0, εd=0.0, V=D / 2)
    sites = siteinds("Electron", N; conserve_qns=true)
    H = MPO(build_opsum(m, sites; phys_sites=collect(1:N), boundary=:full), sites)
    eps = [-D * cos(k * π / (N + 1)) for k in 1:N]
    E_exact = 2 * sum(filter(<(0), eps))                # ×2 for spin
    ψ0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n in 1:N])
    E, _ = dmrg(H, ψ0; nsweeps=12, maxdim=120, cutoff=1e-12, outputlevel=0)
    @test E ≈ E_exact rtol = 1e-6
end
