using ITensorModels: TFIML, build_opsum
using ITensors, ITensorMPS
using ITensors: SiteType
using Random

# At the Ising critical point (J = h_x) with a small longitudinal tilt
# h_z, the TFIML spectrum follows E8 scattering theory (Zamolodchikov
# 1989). In the thermodynamic limit the first two single-particle
# masses satisfy
#
#     m2 / m1 = 2 cos(π/5) = golden ratio φ ≈ 1.618…
#
# This test probes the low-energy sector with ITensorMPS.dmrg_x, which
# targets eigenstates by maximum overlap to an initial MPS. dmrg_x is
# designed for MBL / localised spectra; at a gapless critical point
# the collective low-lying modes are not product-like and dmrg_x is
# not expected to cleanly resolve m1 / m2. The ratio assertion is
# therefore marked @test_broken so the observed value is logged
# without failing CI — flip it to @test once a more suitable solver
# (excited-state DMRG with orthogonal penalty, Lanczos, …) is wired in.

if get(ENV, "ITENSORMODELS_SLOW_TESTS", "false") != "true"
    @info "Skipping TFIML E8 dmrg_x probe (set ITENSORMODELS_SLOW_TESTS=true to enable)"
else
    @testset "TFIML E8 low-energy gap probe via dmrg_x" begin
        N = 16
        m = TFIML(; J=1.0, h_x=1.0, h_z=0.05, site=SiteType("S=1/2"))
        sites = siteinds("S=1/2", N)
        H = MPO(build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)

        rng = MersenneTwister(42)
        ψ0_init = random_mps(rng, sites; linkdims=16)
        sweeps = Sweeps(20)
        maxdim!(sweeps, 10, 20, 40, 60)
        cutoff!(sweeps, 1e-12)
        E0, _ = dmrg(H, ψ0_init, sweeps; outputlevel=0)

        mid = N ÷ 2
        state_up = fill("Up", N)
        flip1 = copy(state_up)
        flip1[mid] = "Dn"
        flip2 = copy(state_up)
        flip2[mid] = "Dn"
        flip2[mid + 1] = "Dn"

        dmrgx_kw = (; nsweeps=20, maxdim=60, cutoff=1e-10, normalize=true, outputlevel=0)
        E1, _ = dmrg_x(H, MPS(sites, flip1); nsite=2, dmrgx_kw...)
        E2, _ = dmrg_x(H, MPS(sites, flip2); nsite=2, dmrgx_kw...)

        Δ1 = E1 - E0
        Δ2 = E2 - E0
        ratio = Δ2 / Δ1
        φ = (1 + sqrt(5)) / 2
        @info "TFIML E8 probe (dmrg_x)" E0 E1 E2 Δ1 Δ2 ratio φ

        # Sanity: both gaps strictly positive.
        @test Δ1 > 0
        @test Δ2 > 0
        # Aspirational: dmrg_x-sampled gap ratio matches the E8 golden-ratio
        # prediction. Broken because dmrg_x targets product-state-overlap
        # eigenstates; the critical TFIML low-lying modes are not
        # product-like. Re-enable once excited-state DMRG is added.
        @test_broken ratio ≈ φ rtol = 0.1
    end
end  # slow-tests guard
