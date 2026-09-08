using ITensorModels
using ITensors
using ITensorMPS
using QAtlas: QAtlas, RiceMele, Energy, OBC
using Random
using Test

# At U = 0 the two spin species decouple and each fills the lower band at half filling,
# so the spinful chain's ground-state energy is twice the spinless Rice-Mele value.
# QAtlas reaches that by diagonalizing the 2N x 2N single-particle matrix; DMRG reaches
# it variationally on an MPS over the full 4^{2N} space. Neither is derived from the other.
#
# QAtlas's OBC(N) is N unit cells, i.e. 2N sites, and its Energy{:per_site} divides by 2N.
function _qatlas_total(v, w, Δ, N)
    return 2 * QAtlas.fetch(RiceMele(; v=v, w=w, Δ=Δ), Energy{:per_site}(), OBC(N)) * (2N)
end

function _dmrg_energy(m, L; seed=7)
    sites = siteinds("Electron", L; conserve_qns=true)
    H = MPO(build_opsum(m, sites; phys_sites=1:L, boundary=:full), sites)
    state = [isodd(k) ? "Up" : "Dn" for k in 1:L]        # half filling, Sz = 0
    ψ0 = random_mps(MersenneTwister(seed), sites, state; linkdims=20)
    sw = Sweeps(24)
    maxdim!(sw, 20, 40, 80, 120, 200)
    cutoff!(sw, 1.0e-14)
    E, _ = dmrg(H, ψ0, sw; outputlevel=0)
    return E
end

@testset "RiceMeleHubbard1D at U=0 is twice the spinless QAtlas chain" begin
    # MEASURED |E_dmrg − 2 E_qatlas| over these fixtures and N ∈ {3,4,6}: 0.0 … 1.1e-10,
    # the worst at the undimerised v = w. rtol = 1e-9 sits ~140x above that.
    for (v, w, Δ) in ((1.0, 0.4, 0.3), (0.4, 1.0, 0.3), (0.7, 1.3, 0.5), (1.0, 1.0, 0.4))
        for N in (3, 4)
            m = RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=0.0)
            @test _dmrg_energy(m, 2N) ≈ _qatlas_total(v, w, Δ, N) rtol = 1.0e-9
        end
    end
end

@testset "on an open chain the Peierls phase is pure gauge" begin
    # A uniform vector potential on an OPEN chain is removable by c_j -> e^{iφ_j} c_j, so it
    # moves no energy — measured on the single-particle matrix at 1e-15 over
    # A ∈ {0.3, 1.0, 2.5}. That makes the OBC energy STRUCTURALLY BLIND to the phase, so it
    # cannot be used to check that the phase is present; it checks something else, and
    # something a sign slip breaks: that the two directions are conjugate. Put e^{-iA} on
    # both and the Hamiltonian stops being Hermitian and the energy moves.
    v, w, Δ, N = 1.0, 0.4, 0.3, 3
    ref = _qatlas_total(v, w, Δ, N)
    for A in (0.0, 0.3, 1.0, 2.5)
        m = RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=0.0, A=A)
        @test _dmrg_energy(m, 2N) ≈ ref rtol = 1.0e-9
    end
    # The interacting chain is gauge-invariant too, and there is no closed form to compare
    # against — so it is compared against ITSELF at another phase, which a non-Hermitian
    # bond would break.
    withU(A) = _dmrg_energy(RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=2.0, A=A), 2N)
    e0 = withU(0.0)
    @test withU(1.3) ≈ e0 rtol = 1.0e-9
    @test e0 > ref                     # U ≠ 0 really did move it, so the above is not trivial
end

@testset "the agreement is a fact about the numbers, not the shape" begin
    # Swapping v and w keeps every dimension and every filling and changes the chain,
    # because OBC puts v on the first bond. MEASURED separation: 1.26 against 1e-13.
    for (v, w, Δ) in ((1.0, 0.4, 0.3), (0.7, 1.3, 0.5))
        N = 4
        E = _dmrg_energy(RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=0.0), 2N)
        @test abs(E - _qatlas_total(v, w, Δ, N)) < 1.0e-9
        @test abs(E - _qatlas_total(w, v, Δ, N)) > 0.1
    end

    # NOT a control here, and deliberately absent: an even-length OBC chain cannot see the
    # sign of Δ. Reversing sites (j -> L+1-j) preserves the bond pattern and maps the +Δ
    # sublattice onto the -Δ one, so H(v,w,Δ) and H(v,w,-Δ) are unitarily equivalent —
    # MEASURED, the two residuals agree bitwise. The operator-level test carries that
    # control instead, where the two Hamiltonians really do differ.

    # U has to move the answer, or the comparison above says nothing about the interaction.
    # MEASURED at (1.0, 0.4, 0.3), L = 8: E = -8.60, -7.58, -5.26, -3.48 for U = 0, 0.5, 2, 4.
    E0 = _dmrg_energy(RiceMeleHubbard1D(; v=1.0, w=0.4, Δ=0.3, U=0.0), 8)
    E2 = _dmrg_energy(RiceMeleHubbard1D(; v=1.0, w=0.4, Δ=0.3, U=2.0), 8)
    @test E2 - E0 > 1.0
end
