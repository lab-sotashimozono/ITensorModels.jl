using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LinearAlgebra: norm
using Test

# The chain the model claims to be, with no half-weights and no bond structure.
function _plain_rice_mele(m, L)
    H = OpSum()
    for i in 1:(L - 1)
        t = isodd(i) ? m.v : m.w
        fwd = iszero(m.A) ? -t : -t * cis(-m.A)
        H += (fwd, "Cdag", i, "C", i + 1)
        H += (conj(fwd), "Cdag", i + 1, "C", i)
        iszero(m.V) || (H += (m.V, "N", i, "N", i + 1))
    end
    for k in 1:L
        H += ((isodd(k) ? m.Δ : -m.Δ), "N", k)
    end
    return H
end

@testset "RiceMele1D: construction" begin
    m = RiceMele1D()
    @test m isa AbstractLatticeModel
    @test (m.v, m.w, m.Δ, m.V, m.A) == (1.0, 1.0, 0.0, 0.0, 0.0)
    @test site_type(m) == SiteType("Fermion")
end

@testset "RiceMele1D: build_opsum is the plain chain" begin
    L = 4
    sites = siteinds("Fermion", L)
    for (v, w, Δ, V, A) in
        ((0.7, 1.3, 0.5, 0.0, 0.0), (1.2, 0.2, -0.4, 0.9, 0.0), (0.6, 0.9, 0.3, 0.0, 0.7))
        m = RiceMele1D(; v=v, w=w, Δ=Δ, V=V, A=A)
        got = prod(MPO(build_opsum(m, sites; phys_sites=1:L, boundary=:full), sites))
        want = prod(MPO(_plain_rice_mele(m, L), sites))
        @test norm(got - want) < 1.0e-12 * max(norm(want), 1.0)
    end
end

@testset "RiceMele1D: the comparison above can fail" begin
    L = 4
    sites = siteinds("Fermion", L)
    m = RiceMele1D(; v=0.7, w=1.3, Δ=0.5, V=0.9)
    got = prod(MPO(build_opsum(m, sites; phys_sites=1:L, boundary=:full), sites))
    for wrong in (
        RiceMele1D(; v=1.3, w=0.7, Δ=0.5, V=0.9),    # v <-> w
        RiceMele1D(; v=0.7, w=1.3, Δ=-0.5, V=0.9),   # sublattice flipped
        RiceMele1D(; v=0.7, w=1.3, Δ=0.5, V=0.0),    # no interaction
    )
        @test norm(got - prod(MPO(_plain_rice_mele(wrong, L), sites))) > 0.1
    end
end

@testset "RiceMele1D: Δ = 0 is SSH1D" begin
    # The staggered potential is the ONLY difference, so at Δ = 0 the two must agree —
    # which pins that v and w land on the same bonds SSH1D's t1 and t2 do.
    L = 4
    sites = siteinds("Fermion", L)
    a = prod(
        MPO(
            build_opsum(RiceMele1D(; v=0.8, w=1.0), sites; phys_sites=1:L, boundary=:full),
            sites,
        ),
    )
    b = prod(
        MPO(
            build_opsum(SSH1D(; t1=0.8, t2=1.0), sites; phys_sites=1:L, boundary=:full),
            sites,
        ),
    )
    @test norm(a - b) < 1.0e-12
    # …and Δ ≠ 0 really does separate them, so the line above is not comparing two zeros.
    c = prod(
        MPO(
            build_opsum(
                RiceMele1D(; v=0.8, w=1.0, Δ=0.3), sites; phys_sites=1:L, boundary=:full
            ),
            sites,
        ),
    )
    @test norm(c - b) > 0.1
end

@testset "RiceMele1D: a forward-only split reassembles bond_term" begin
    # This is the contract an iTEBD driver relies on: it needs the FORWARD hopping block
    # alone so it can attach a Peierls phase, and the rest of the bond separately. Their sum
    # has to be the model's own `bond_term`, or the driver is evolving a different chain.
    v, w, Δ = 0.75, 0.25, 0.1
    sites = siteinds("Fermion", 3)
    m = RiceMele1D(; v=v, w=w, Δ=Δ)
    for n in 1:2
        t = isodd(n) ? v : w
        oh = OpSum()
        oh += -t, "Cdag", n, "C", n + 1
        oo = OpSum()
        oo += (isodd(n) ? Δ : -Δ) / 2, "N", n
        oo += (isodd(n + 1) ? Δ : -Δ) / 2, "N", n + 1
        K = prod(MPO(oh, sites))
        split = K + swapprime(dag(K), 0 => 1) + prod(MPO(oo, sites))
        @test norm(split - prod(MPO(bond_term(m, n, n + 1), sites))) < 1.0e-12
    end
end

@testset "RiceMele1D: the split protocol adds back up" begin
    # `bond_coupling_term` / `onsite_term` are the split the modulation wrapper consumes;
    # `bond_term` / `boundary_patch` are the assembled form `build_opsum` walks. Nothing else
    # in this file reaches `onsite_term`, so a wrong one would ship.
    L = 2
    sites = siteinds("Fermion", L)
    mpo(H) = prod(MPO(H, sites))
    for (v, w, Δ, V) in ((0.7, 1.3, 0.5, 0.9), (1.2, 0.2, -0.4, 0.0))
        m = RiceMele1D(; v=v, w=w, Δ=Δ, V=V)
        for k in 1:L
            @test norm(2 * mpo(boundary_patch(m, k)) - mpo(onsite_term(m, k))) < 1.0e-12
        end
        assembled = mpo(bond_term(m, 1, 2))
        split =
            mpo(bond_coupling_term(m, 1, 2)) +
            mpo(boundary_patch(m, 1)) +
            mpo(boundary_patch(m, 2))
        @test norm(assembled - split) < 1.0e-12
        # Not vacuous: the two sites carry OPPOSITE staggered potentials, so giving both
        # halves to the same end has to break it whenever Δ ≠ 0.
        if !iszero(Δ)
            crossed = mpo(bond_coupling_term(m, 1, 2)) + 2 * mpo(boundary_patch(m, 1))
            @test norm(assembled - crossed) > 0.1 * abs(Δ)
        end
    end
end

@testset "RiceMele1D: an embedding that would flip the parities is refused" begin
    sites = siteinds("Fermion", 6)
    m = RiceMele1D(; v=1.0, w=0.2, Δ=0.5)
    @test build_opsum(m, sites; phys_sites=1:4, boundary=:full) isa OpSum
    @test build_opsum(m, sites; phys_sites=3:6, boundary=:full) isa OpSum
    @test_throws ArgumentError build_opsum(m, sites; phys_sites=2:5, boundary=:full)
    @test_throws ArgumentError build_opsum(m, sites; phys_sites=1:2:5, boundary=:full)
end

@testset "RiceMele1D: onsite_observable_op matches its spinless siblings" begin
    m = RiceMele1D()
    for name in (:n, :c, :cdag)
        @test onsite_observable_op(m, name) == onsite_observable_op(TightBinding1D(), name)
    end
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end
