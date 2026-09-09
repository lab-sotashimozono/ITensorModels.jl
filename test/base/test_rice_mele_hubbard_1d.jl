using ITensorModels
using ITensors
using ITensors: SiteType
using ITensorMPS
using LinearAlgebra: norm
using Test

# The chain the model claims to be, written out with no half-weights and no bond
# structure. Comparing operators rather than energies: an energy can agree while the
# two end sites are wrong, and this cannot.
function _plain_rice_mele_hubbard(m, L; edge_weight=1.0)
    H = OpSum()
    for i in 1:(L - 1)
        t = isodd(i) ? m.v : m.w
        H += -t, "Cdagup", i, "Cup", i + 1
        H += -t, "Cdagup", i + 1, "Cup", i
        H += -t, "Cdagdn", i, "Cdn", i + 1
        H += -t, "Cdagdn", i + 1, "Cdn", i
    end
    for k in 1:L
        c = (k == 1 || k == L) ? edge_weight : 1.0
        H += c * m.U, "Nupdn", k
        H += c * (isodd(k) ? m.Δ : -m.Δ), "Ntot", k
    end
    return H
end

@testset "RiceMeleHubbard1D: construction" begin
    m = RiceMeleHubbard1D()
    @test m isa AbstractLatticeModel
    @test (m.v, m.w, m.Δ, m.U) == (1.0, 1.0, 0.0, 0.0)
    @test site_type(m) == SiteType("Electron")
end

@testset "RiceMeleHubbard1D: the hopping alternates with bond parity" begin
    m = RiceMeleHubbard1D(; v=0.3, w=1.7)
    for (i, want) in ((1, -0.3), (2, -1.7), (3, -0.3), (4, -1.7))
        terms = collect(ITensors.terms(bond_coupling_term(m, i, i + 1)))
        @test length(terms) == 4
        @test all(t -> ITensors.coefficient(t) ≈ want, terms)
    end
end

@testset "RiceMeleHubbard1D: build_opsum is the plain chain" begin
    L = 4
    sites = siteinds("Electron", L)
    for (v, w, Δ, U) in ((0.7, 1.3, 0.5, 3.0), (1.2, 0.2, -0.4, 0.0), (1.0, 1.0, 0.0, 2.0))
        m = RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=U)
        for (boundary, edge) in ((:full, 1.0), (:bulk_half_edge, 0.5))
            got = prod(MPO(build_opsum(m, sites; phys_sites=1:L, boundary=boundary), sites))
            want = prod(MPO(_plain_rice_mele_hubbard(m, L; edge_weight=edge), sites))
            @test norm(got - want) < 1.0e-12 * max(norm(want), 1.0)
        end
    end
end

@testset "RiceMeleHubbard1D: the comparison above can fail" begin
    # Same shape, one parameter wrong. Without this the test above passes for a model
    # that ignores its arguments.
    L = 4
    sites = siteinds("Electron", L)
    m = RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.5, U=3.0)
    got = prod(MPO(build_opsum(m, sites; phys_sites=1:L, boundary=:full), sites))
    for wrong in (
        RiceMeleHubbard1D(; v=1.3, w=0.7, Δ=0.5, U=3.0),   # v <-> w
        RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=-0.5, U=3.0),  # the sublattice flipped
        RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.5, U=0.0),   # no interaction
    )
        want = prod(MPO(_plain_rice_mele_hubbard(wrong, L), sites))
        @test norm(got - want) > 0.1
    end
end

@testset "RiceMeleHubbard1D: the split protocol adds back up" begin
    # `bond_coupling_term` / `onsite_term` are the split the modulation wrapper consumes, and
    # `bond_term` / `boundary_patch` are the assembled form. The interface's contract is that
    # the on-site weight is halved onto each endpoint, so the two have to reconcile exactly:
    # a bond term is its coupling plus half the on-site weight at each end, and two boundary
    # patches make one whole on-site term. Nothing else in the suite reaches `onsite_term`.
    L = 2
    sites = siteinds("Electron", L)
    mpo(H) = prod(MPO(H, sites))
    for (v, w, Δ, U) in ((0.7, 1.3, 0.5, 3.0), (1.2, 0.2, -0.4, 0.0), (1.0, 1.0, 0.0, 2.0))
        m = RiceMeleHubbard1D(; v=v, w=w, Δ=Δ, U=U)
        for k in 1:L
            half = mpo(boundary_patch(m, k))
            @test norm(2 * half - mpo(onsite_term(m, k))) < 1.0e-12
        end
        assembled = mpo(bond_term(m, 1, 2))
        split =
            mpo(bond_coupling_term(m, 1, 2)) +
            mpo(boundary_patch(m, 1)) +
            mpo(boundary_patch(m, 2))
        @test norm(assembled - split) < 1.0e-12
        # Not vacuous: the two sites carry OPPOSITE staggered potentials, so swapping which
        # end gets which half has to break it whenever Δ ≠ 0.
        if !iszero(Δ)
            crossed = mpo(bond_coupling_term(m, 1, 2)) + 2 * mpo(boundary_patch(m, 1))
            @test norm(assembled - crossed) > 0.1 * abs(Δ)
        end
    end
end

@testset "RiceMeleHubbard1D: A = 0 leaves the model real" begin
    # A phase of zero must not turn the coefficients complex: the field-free model is the
    # one every other test and the DMRG cross-check build, and a complex MPO carrying zero
    # phase would change their arithmetic for nothing.
    m0 = RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.5, U=3.0)
    @test m0.A == 0.0
    for t in ITensors.terms(bond_coupling_term(m0, 1, 2))
        @test ITensors.coefficient(t) isa Real || imag(ITensors.coefficient(t)) == 0
    end
    sites = siteinds("Electron", 4)
    H0 = MPO(build_opsum(m0, sites; phys_sites=1:4, boundary=:full), sites)
    @test eltype(prod(H0)) <: Real
end

@testset "RiceMeleHubbard1D: the Peierls phase is conjugate across the bond" begin
    # Hermiticity is what makes it a gauge rather than a gain/loss term, and it is the one
    # thing a sign slip in `_hop_amplitudes` destroys.
    A = 0.7
    m = RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.5, U=3.0, A=A)
    fwd, bwd = ITensorModels._hop_amplitudes(m, 1, 2)
    @test bwd ≈ conj(fwd)
    @test abs(fwd) ≈ 0.7
    @test angle(-fwd) ≈ -A            # `c†_i c_{i+1}` carries e^{-iA}: this fixes H_k -> H_{k-A}
    sites = siteinds("Electron", 4)
    H = prod(MPO(build_opsum(m, sites; phys_sites=1:4, boundary=:full), sites))
    @test norm(H - swapprime(dag(H), 0 => 1)) < 1.0e-12 * norm(H)
    # …and it is not the zero phase in disguise.
    H0 = prod(
        MPO(
            build_opsum(
                RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.5, U=3.0),
                sites;
                phys_sites=1:4,
                boundary=:full,
            ),
            sites,
        ),
    )
    @test norm(H - H0) > 0.1
end

@testset "RiceMeleHubbard1D: an embedding that would flip the parities is refused" begin
    # Every other test here builds with `phys_sites = 1:L`, where the label parity and the
    # chain position agree — so none of them can see that this model reads the dimerisation
    # off the LABEL. MEASURED at (1.0, 0.2, 0.5) on four sites: ‖H‖ = 102.0 for `1:4` against
    # 80.4 for `2:5`, the same chain shifted by one site. SSH1D, which reads positions,
    # returns 8.079604 for both. Refused rather than silently answered.
    sites = siteinds("Electron", 6)
    m = RiceMeleHubbard1D(; v=1.0, w=0.2, Δ=0.5, U=0.0)
    @test build_opsum(m, sites; phys_sites=1:4, boundary=:full) isa OpSum
    @test build_opsum(m, sites; phys_sites=3:6, boundary=:full) isa OpSum   # odd start
    @test_throws ArgumentError build_opsum(m, sites; phys_sites=2:5, boundary=:full)
    @test_throws ArgumentError build_opsum(m, sites; phys_sites=1:2:5, boundary=:full)
    # An odd-start consecutive run really is the same Hamiltonian, not merely accepted.
    a = prod(MPO(build_opsum(m, sites; phys_sites=1:4, boundary=:full), sites))
    b = prod(MPO(build_opsum(m, sites; phys_sites=3:6, boundary=:full), sites))
    @test norm(a) ≈ norm(b)
end

@testset "RiceMeleHubbard1D: onsite_observable_op" begin
    # The same site type as Hubbard1D, so the same names have to work: a consumer that
    # asks an Electron model for `:sz` should not care which Electron model it got.
    m = RiceMeleHubbard1D()
    for name in (:nup, :ndn, :n, :nupdn, :sz)
        @test onsite_observable_op(m, name) == onsite_observable_op(Hubbard1D(), name)
    end
    @test onsite_observable_op(m, :n) == "Ntot"
    @test_throws ErrorException onsite_observable_op(m, :bogus)
end
