using ITensorModels
using ITensors
using ITensors: SiteType, array, prime
using ITensorMPS
using LinearAlgebra: I, eigvals, kron
using Random
using Random: Xoshiro
using Statistics: cor, mean
using Test

# `-Σ J_i Z_i Z_{i+1} - Σ h_i X_i` by hand. `scale` is the local operator
# normalisation: 1 for Pauli `Z`/`X`, 1/2 for spin `Sz`/`Sx`.
function dense_random_tfim(J, h; scale=1.0)
    L = length(h)
    σx = scale .* Float64[0 1; 1 0]
    σz = scale .* Float64[1 0; 0 -1]
    id(n) = Matrix{Float64}(I, 1 << n, 1 << n)
    op(o, k) = kron(id(k - 1), o, id(L - k))
    H = zeros(Float64, 1 << L, 1 << L)
    for i in 1:(L - 1)
        H .-= J[i] .* (op(σz, i) * op(σz, i + 1))
    end
    for k in 1:L
        H .-= h[k] .* op(σx, k)
    end
    return H
end

# `reshape` makes the first index vary fastest, so the site order is reversed to
# put site 1 in the most significant bit.
function dense_of(mpo, sites)
    L = length(sites)
    t = reduce(*, mpo)
    order = reverse(1:L)
    return reshape(
        array(t, [sites[i] for i in order]..., [prime(sites[i]) for i in order]...),
        1 << L,
        1 << L,
    )
end

function dense_mpo(m, sites; phys_sites=collect(1:length(sites)))
    os = build_opsum(m, sites; phys_sites=phys_sites, boundary=:full)
    return dense_of(MPO(os, sites), sites)
end

@testset "RandomTFIM: construction and shape" begin
    m = RandomTFIM(; J=[0.3, 0.7], h=[1.0, 2.0, 0.5])
    @test m isa AbstractLatticeModel
    @test site_type(m) == SiteType("S=1/2")        # the family default
    @test_throws ErrorException RandomTFIM(; J=[0.3], h=[1.0, 2.0, 0.5])
    # L bonds on L sites is the periodic chain.
    @test_throws ErrorException RandomTFIM(; J=[0.3, 0.7, 0.1], h=[1.0, 2.0, 0.5])
    @test_throws ErrorException RandomTFIM(; J=Float64[], h=[1.0])
end

@testset "random_tfim: the seed fixes the couplings" begin
    a = random_tfim(; L=12, seed=7)
    b = random_tfim(; L=12, seed=7)
    @test a.J == b.J && a.h == b.h
    @test random_tfim(; L=12, seed=8).J != a.J
    @test_throws ErrorException random_tfim(; L=10, seed=1, D=0.0)
    @test_throws ErrorException random_tfim(; L=1, seed=1)
    # Xoshiro rejects a negative seed on the declared Julia floor.
    @test_throws ErrorException random_tfim(; L=10, seed=-1)
end

@testset "random_tfim: J and h are independent draws from one distribution" begin
    L = 4000
    m = random_tfim(; L=L, seed=1)
    @test 0.48 < mean(m.J) < 0.52
    # Independence is the content of "J and h share a distribution": a chain that
    # reused one draw for both would pass every moment test and be a different
    # model (h ≡ J is not the critical ensemble, it is one chain).
    @test abs(cor(m.J, m.h[1:(L - 1)])) < 0.1
    # Same distribution. sd of the difference of log-means is sqrt(2/L) = 0.022,
    # so the gate is 5σ — a tighter one is a coin flip on an unrelated upgrade.
    @test abs(mean(log, m.J) - mean(log, m.h)) < 5 * sqrt(2 / L)

    # D scales the spread of the logarithms, in BOTH couplings.
    m2 = random_tfim(; L=L, seed=1, D=2.0)
    @test mean(log, m2.J) ≈ 2 * mean(log, m.J) rtol = 1e-12
    @test mean(log, m2.h) ≈ 2 * mean(log, m.h) rtol = 1e-12

    # An AbstractRNG is accepted, and is advanced — the documented behaviour.
    rng = Xoshiro(5)
    first = random_tfim(; L=8, seed=rng)
    second = random_tfim(; L=8, seed=rng)
    @test first.J != second.J
    @test first.J == random_tfim(; L=8, seed=Xoshiro(5)).J
end

@testset "rescale_disorder is the continuation step" begin
    m = random_tfim(; L=20, seed=3)
    @test rescale_disorder(m, 1).J == m.J
    @test rescale_disorder(m, 1).h == m.h
    @test all(≈(1.0), rescale_disorder(m, 0).J)        # α = 0 is the clean chain
    @test all(≈(1.0), rescale_disorder(m, 0).h)
    # Both couplings carry the SAME exponent — the reason criticality survives.
    @test rescale_disorder(m, 0.5).J ≈ sqrt.(m.J)
    @test rescale_disorder(m, 0.5).h ≈ sqrt.(m.h)
    @test mean(log, rescale_disorder(m, 0.4).h) ≈ 0.4 * mean(log, m.h)
    # The realisation is kept: the couplings do not change order.
    @test sortperm(rescale_disorder(m, 0.3).J) == sortperm(m.J)
    @test site_type(rescale_disorder(m, 0.3)) == site_type(m)
    @test_throws ErrorException rescale_disorder(m, -1)
    # `J^α` is not a disorder strength if a coupling is negative, and `^` would
    # throw from inside a broadcast rather than from the model.
    @test_throws ErrorException rescale_disorder(RandomTFIM(; J=[-1.0], h=[1.0, 1.0]), 0.5)
end

@testset "RandomTFIM: the MPO is the Hamiltonian it says it is" begin
    for (L, seed) in ((3, 1), (4, 2), (6, 3))
        m = random_tfim(; L=L, seed=seed, site=SiteType("Qubit"))
        sites = siteinds("Qubit", L)
        @test dense_mpo(m, sites) ≈ dense_random_tfim(m.J, m.h) atol = 1e-12
    end
    # The site type reaches the Hamiltonian: on S=1/2 the operators are halved,
    # so hardcoding "Z"/"X" in the term methods would be a 4× bond error here.
    for (L, seed) in ((3, 4), (5, 5))
        m = random_tfim(; L=L, seed=seed)
        @test site_type(m) == SiteType("S=1/2")
        sites = siteinds("S=1/2", L)
        @test dense_mpo(m, sites) ≈ dense_random_tfim(m.J, m.h; scale=0.5) atol = 1e-12
    end
end

@testset "RandomTFIM: constant couplings reduce to TFIM, at the defaults" begin
    L, J, h = 6, 0.8, 1.3
    sites = siteinds("S=1/2", L)
    @test dense_mpo(RandomTFIM(; J=fill(J, L - 1), h=fill(h, L)), sites) ≈
        dense_mpo(TFIM(; J=J, h=h), sites) atol = 1e-12
end

@testset "RandomTFIM: couplings follow the ordinal, not the MPS position" begin
    L = 4
    m = random_tfim(; L=L, seed=9, site=SiteType("Qubit"))
    plain = dense_mpo(m, siteinds("Qubit", L))

    # An offset window — ThermalMPS's aux chain layout. The chain must be the same
    # realisation sitting further along, i.e. H ⊗ nothing on the spectators.
    sites = siteinds("Qubit", L + 2)
    offset = dense_mpo(m, sites; phys_sites=collect(2:(L + 1)))
    id2 = Matrix{Float64}(I, 2, 2)
    @test offset ≈ kron(id2, plain, id2) atol = 1e-12

    # An interleaved window — the purification layout. Same spectrum, each level
    # repeated once per spectator.
    inter_sites = siteinds("Qubit", 2L)
    inter = dense_mpo(m, inter_sites; phys_sites=collect(1:2:(2L - 1)))
    @test sort(unique(round.(real.(eigvals(inter)); digits=10))) ≈
        sort(unique(round.(real.(eigvals(plain)); digits=10))) atol = 1e-8

    # A window is not the chain.
    @test_throws ErrorException local_ham_terms(m, 1:(L - 1))
    @test_throws ErrorException local_ham_terms(m, 1:(L + 1))
    @test_throws ErrorException local_ham_terms(m, 1:L; boundary=:nonsense)
end

@testset "RandomTFIM: onsite observables" begin
    m = random_tfim(; L=4, seed=1, site=SiteType("Qubit"))
    @test onsite_observable_op(m, :sx) == "X"
    @test onsite_observable_op(m, :sz) == "Z"
    @test onsite_observable_op(m, :sy) == "Y"
    @test_throws ErrorException onsite_observable_op(m, :nonsense)
    @test onsite_observable_op(random_tfim(; L=4, seed=1), :sx) == "Sx"
end

@testset "RandomTFIM: the split protocol modulated consumes" begin
    m = random_tfim(; L=5, seed=6, site=SiteType("Qubit"))
    sites = siteinds("Qubit", 5)
    wrapped = modulated(m; L=5, modulation=Uniform())
    @test dense_mpo(wrapped, sites) ≈ dense_mpo(m, sites) atol = 1e-12
    @test length(bond_coupling_term(m, 2, 3)) == 1
    @test length(onsite_term(m, 4)) == 1
end

@testset "RandomTFIM: bond_term reads position as ordinal, and says so" begin
    m = random_tfim(; L=6, seed=4)
    @test_throws ErrorException bond_term(m, 1, 3)     # not nearest neighbour
    @test_throws ErrorException bond_term(m, 3, 2)     # descending
    @test_throws ErrorException bond_term(m, 6, 7)     # past the last bond
    @test_throws ErrorException onsite_term(m, 7)
    @test_throws ErrorException boundary_patch(m, 0)
end
