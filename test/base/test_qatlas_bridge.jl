using ITensorModels: TFIM, XXZ1D, Heisenberg1D, site_type, to_qatlas, from_qatlas
using ITensors: SiteType, OpSum
using QAtlas: QAtlas
using QAtlas: Energy, MassGap, Infinite, OBC

# These tests are deliberately non-tautological: they validate the
# QAtlas bridge against (a) analytic TFIM results and (b) agreement
# between the ITensorModels + ext forwarder and a direct QAtlas call on
# a hand-written struct.

@testset "to_qatlas round-trip" begin
    m_q = TFIM(; J=0.4, h=0.6, site=SiteType("Qubit"))
    @test to_qatlas(m_q).J == 0.4 && to_qatlas(m_q).h == 0.6

    back = from_qatlas(to_qatlas(m_q))
    @test back.J == 0.4 && back.h == 0.6 && back.site === SiteType("Qubit")
end

@testset "MassGap matches analytic 2|h-J|" begin
    for (J, h) in [(1.0, 0.3), (1.0, 1.7), (0.8, 0.5)]
        m = TFIM(; J=J, h=h, site=SiteType("Qubit"))
        Δ = QAtlas.fetch(m, MassGap(), Infinite())
        @test Δ ≈ 2 * abs(h - J) rtol = 1e-12
    end
end

@testset "SiteType unit conversion: S=1/2 ≡ Qubit(J/4, h/2)" begin
    # Same physical Hamiltonian expressed in two conventions.
    J, h = 1.0, 0.5
    m_s = TFIM(; J=J, h=h)                                 # S=1/2 default
    m_q = TFIM(; J=J / 4, h=h / 2, site=SiteType("Qubit"))  # equivalent Pauli

    @test QAtlas.fetch(m_s, Energy(), Infinite(); beta=2.0) ≈
        QAtlas.fetch(m_q, Energy(), Infinite(); beta=2.0) rtol = 1e-12
    @test QAtlas.fetch(m_s, MassGap(), Infinite()) ≈
        QAtlas.fetch(m_q, MassGap(), Infinite()) rtol = 1e-12
end

@testset "Forwarder parity with direct QAtlas call" begin
    # Hand-written QAtlas struct with the Pauli couplings that match a
    # :S=1/2 ITensorModels.TFIM(J=1, h=0.5). The forwarder's output must
    # equal a direct fetch on the Pauli struct.
    m = TFIM(; J=1.0, h=0.5)
    qm_direct = QAtlas.TFIM(; J=0.25, h=0.25)

    @test QAtlas.fetch(m, Energy(), OBC(12); beta=3.0) ≈
        QAtlas.fetch(qm_direct, Energy(), OBC(12); beta=3.0) rtol = 1e-12
    @test QAtlas.fetch(m, MassGap(), OBC(12)) ≈ QAtlas.fetch(qm_direct, MassGap(), OBC(12)) rtol =
        1e-12
end

@testset "to_qatlas undefined for unsupported SiteType" begin
    m_s1 = TFIM(; site=SiteType("S=1"))
    @test_throws MethodError to_qatlas(m_s1)
end

# ── XXZ1D / Heisenberg1D round-trip ──────────────────────────────────

@testset "XXZ1D round-trip preserves J and Δ" begin
    # XXZ1D uses spin-½ coefficients on both sides, so to_qatlas /
    # from_qatlas pass (J, Δ) through unchanged.
    m = XXZ1D(; J=0.7, Δ=-0.3)
    qm = to_qatlas(m)
    @test qm.J == 0.7 && qm.Δ == -0.3
    back = from_qatlas(qm)
    @test back.J == 0.7 && back.Δ == -0.3
end

@testset "Heisenberg1D round-trip — J = 1 only" begin
    # `QAtlas.Heisenberg1D` carries no J field; to_qatlas drops it and
    # from_qatlas restores J = 1. So a J = 1 round-trip is exact, but
    # a J ≠ 1 round-trip drops J (documented). Tests both.
    m1 = Heisenberg1D(; J=1.0)
    @test from_qatlas(to_qatlas(m1)).J == 1.0

    m2 = Heisenberg1D(; J=0.7)
    @test from_qatlas(to_qatlas(m2)).J == 1.0
    @test m2.J != from_qatlas(to_qatlas(m2)).J
end

# ── site_type propagation through the bridge ─────────────────────────

@testset "site_type accessor reflects the model's site field" begin
    @test site_type(TFIM(; site=SiteType("S=1/2"))) === SiteType("S=1/2")
    @test site_type(TFIM(; site=SiteType("Qubit"))) === SiteType("Qubit")
    @test site_type(XXZ1D(; site=SiteType("S=1/2"))) === SiteType("S=1/2")
    @test site_type(XXZ1D(; site=SiteType("Qubit"))) === SiteType("Qubit")
    @test site_type(Heisenberg1D(; site=SiteType("Qubit"))) === SiteType("Qubit")
end

@testset "from_qatlas defaults to the canonical site (Pauli for TFIM, S=½ for XXZ)" begin
    # `QAtlas.TFIM(J, h)` is in Pauli units, so the inverse map lands
    # on SiteType("Qubit") to keep coefficients identity-preserving.
    @test site_type(from_qatlas(QAtlas.TFIM(; J=0.4, h=0.6))) === SiteType("Qubit")
    # XXZ1D / Heisenberg1D bridge preserves spin-½ units → S=1/2 default.
    @test site_type(from_qatlas(QAtlas.XXZ1D(; J=1.0, Δ=0.5))) === SiteType("S=1/2")
    @test site_type(from_qatlas(QAtlas.Heisenberg1D())) === SiteType("S=1/2")
end

# ── bond_term emits operator names matching the SiteType ─────────────
#
# Verifies that XXZ1D's `bond_term` picks up the right operator
# strings ("Sα" vs "X/Y/Z") through the `_xxz_ops(site)` dispatch:
# the same physical Hamiltonian, expressed on either SiteType, must
# act identically on a basis MPS.  A regression here (e.g. the wrong
# site type silently keeps emitting `Sα`) would manifest in the
# downstream MPO building as either a missing-op error or a wrong
# inner product — this test catches the latter (silent) failure mode.

@testset "bond_term: S=1/2 and Qubit site types encode the same H" begin
    using ITensorModels: bond_term
    using ITensorMPS: MPO, MPS, inner
    using ITensorSiteKit: PhysInds

    N = 4
    sites_s = PhysInds(N; SiteType="S=1/2")
    sites_q = PhysInds(N; SiteType="Qubit")
    m_s = XXZ1D(; J=1.0, Δ=0.5, site=SiteType("S=1/2"))
    m_q = XXZ1D(; J=1.0, Δ=0.5, site=SiteType("Qubit"))

    os_s = OpSum()
    os_q = OpSum()
    for j in 1:(N - 1)
        os_s += bond_term(m_s, j, j + 1)
        os_q += bond_term(m_q, j, j + 1)
    end
    H_s = MPO(os_s, sites_s)
    H_q = MPO(os_q, sites_q)

    # ⟨Up|H|Up⟩ — only the Δ Sᶻ Sᶻ term survives. With N - 1 bonds
    # and Sᶻ-eigenvalue (1/2) per site, this equals
    # J Δ (N - 1) × (1/4) on either SiteType (the `scale = 1/4`
    # rescale in `_xxz_ops(::Qubit)` exactly compensates `Z² = +1`
    # vs `Sᶻ² = 1/4`).  A silent site-type → operator-name mismatch
    # would land here.
    ψ_s_up = MPS(sites_s, "Up")
    ψ_q_up = MPS(sites_q, "Up")
    expected = m_s.J * m_s.Δ * (N - 1) / 4
    @test real(inner(ψ_s_up', H_s, ψ_s_up)) ≈ expected rtol = 1e-12
    @test real(inner(ψ_q_up', H_q, ψ_q_up)) ≈ expected rtol = 1e-12
end


@testset "S1Heisenberg1D bridge: to_qatlas + round-trip" begin
    m = S1Heisenberg1D(; J=1.3)
    qm = to_qatlas(m)
    @test qm isa QAtlas.S1Heisenberg1D
    @test qm.J ≈ 1.3
    back = from_qatlas(qm)
    @test back isa S1Heisenberg1D
    @test back.J ≈ 1.3
end

@testset "S1Heisenberg1D: forwarder routes Energy fetch through QAtlas" begin
    # Compare the forwarded fetch on the ITensorModels struct against a
    # direct fetch on a hand-built QAtlas struct -- they must agree
    # bitwise modulo float rounding.
    m = S1Heisenberg1D(; J=1.0)
    qm_direct = QAtlas.S1Heisenberg1D(; J=1.0)
    N = 6
    @test QAtlas.fetch(m, Energy(), OBC(N); beta=10.0) ≈
        QAtlas.fetch(qm_direct, Energy(), OBC(N); beta=10.0) rtol = 1e-12
end

@testset "S1Heisenberg1D: DMRG GS energy matches QAtlas dense ED (low-T limit)" begin
    # High-beta thermal energy from QAtlas dense ED is the ground-state
    # energy. Compare against DMRG on the same OBC chain.
    using ITensorMPS
    using Random
    using ITensors: MPO, siteinds
    using ITensorMPS: random_mps, dmrg, Sweeps, maxdim!, cutoff!

    N = 6
    J = 1.0
    m = S1Heisenberg1D(; J=J)
    E_qatlas = QAtlas.fetch(m, Energy(), OBC(N); beta=50.0)

    sites = siteinds("S=1", N)
    H = MPO(ITensorModels.build_opsum(m, sites; phys_sites=1:N, boundary=:full), sites)
    psi0 = random_mps(MersenneTwister(0x51), sites; linkdims=8)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 40, 60, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100)
    cutoff!(sweeps, 1e-12)
    E_dmrg, _ = dmrg(H, psi0, sweeps; outputlevel=0)

    @test E_dmrg ≈ E_qatlas rtol = 1e-5
end
