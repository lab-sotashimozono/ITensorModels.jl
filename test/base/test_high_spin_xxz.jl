# High-spin XXZ1D support (S=3/2, S=2), enabled by the SiteType registration in
# ITensorSiteKit ≥ v0.2.0 (Models #18). Checks that XXZ1D builds and DMRGs on a
# high-spin site, validated against the INDEPENDENT closed-form Heisenberg-dimer
# ground state E₀ = −J·S(S+1) (two spin-S coupled by S₁·S₂; total-spin-0 singlet).

using ITensorModels, Test
using ITensors: SiteType, dim
using ITensorMPS: dmrg, random_mps, siteinds, MPO

@testset "XXZ1D high-spin $st (Δ=1 Heisenberg dimer GS = -S(S+1))" for (st, S) in (
    ("S=3/2", 1.5), ("S=2", 2.0)
)
    m = XXZ1D(; J=1.0, Δ=1.0, site=SiteType(st))
    @test site_type(m) == SiteType(st)

    # two-site chain; H = S₁·S₂ ⇒ singlet GS energy = -S(S+1)
    sites = siteinds(st, 2)
    @test all(dim.(sites) .== Int(2S + 1))
    H = MPO(build_opsum(m, sites; phys_sites=1:2, boundary=:full), sites)
    ψ0 = random_mps(sites; linkdims=4)
    E, _ = dmrg(H, ψ0; nsweeps=10, maxdim=[10, 20, 40, 40], cutoff=1e-13, outputlevel=0)
    @test E ≈ -S * (S + 1) atol = 1e-6
end

@testset "S1Heisenberg1D still routes through XXZ1D (regression)" begin
    m = S1Heisenberg1D(; J=1.0)
    @test site_type(m) == SiteType("S=1")
    sites = siteinds("S=1", 2)
    H = MPO(build_opsum(m, sites; phys_sites=1:2, boundary=:full), sites)
    E, _ = dmrg(
        H, random_mps(sites; linkdims=4); nsweeps=8, maxdim=40, cutoff=1e-12, outputlevel=0
    )
    @test E ≈ -2.0 atol = 1e-6   # S=1 dimer: -S(S+1) = -2
end
