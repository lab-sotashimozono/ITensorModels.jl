using ITensorModels
using ITensors
using ITensors: SiteType, array, prime
using ITensorMPS
using LinearAlgebra: I, kron
using Random
using Statistics: mean
using Test

# `-Σ J_i Z_i Z_{i+1} - Σ h_i X_i` written out by hand, site 1 the most
# significant Kronecker factor.
function dense_random_tfim(J, h)
    L = length(h)
    σx = Float64[0 1; 1 0]
    σz = Float64[1 0; 0 -1]
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

# `reshape` makes the FIRST index vary fastest, so the site order handed to
# `array` is reversed to put site 1 in the most significant bit — the
# convention `dense_random_tfim` builds with.
function dense_mpo(m, sites)
    L = length(sites)
    t = reduce(
        *, MPO(build_opsum(m, sites; phys_sites=collect(1:L), boundary=:full), sites)
    )
    order = reverse(1:L)
    return reshape(
        array(t, [sites[i] for i in order]..., [prime(sites[i]) for i in order]...),
        1 << L,
        1 << L,
    )
end

@testset "RandomTFIM: construction and shape" begin
    m = RandomTFIM([0.3, 0.7], [1.0, 2.0, 0.5])
    @test m isa AbstractLatticeModel
    @test nsites(m) == 3
    @test site_type(m) == SiteType("Qubit")
    @test_throws ErrorException RandomTFIM([0.3], [1.0, 2.0, 0.5])
    # L bonds on L sites is the periodic chain.
    @test_throws ErrorException RandomTFIM([0.3, 0.7, 0.1], [1.0, 2.0, 0.5])
    @test_throws ErrorException RandomTFIM(Float64[], [1.0])
end

@testset "RandomTFIM: a seed fixes the couplings" begin
    a = RandomTFIM(; L=12, seed=7)
    b = RandomTFIM(; L=12, seed=7)
    @test a.J == b.J && a.h == b.h
    @test RandomTFIM(; L=12, seed=8).J != a.J

    # D = 1 is the uniform ensemble, and J and h share it: the critical line.
    m = RandomTFIM(; L=4000, seed=1)
    @test 0.48 < mean(m.J) < 0.52
    @test abs(mean(log, m.J) - mean(log, m.h)) < 0.05
    # D scales the spread of the logarithms and nothing else.
    m2 = RandomTFIM(; L=4000, seed=1, D=2.0)
    @test mean(log, m2.J) ≈ 2 * mean(log, m.J) rtol = 1e-12
    @test_throws ErrorException RandomTFIM(; L=10, seed=1, D=0.0)
    @test_throws ErrorException RandomTFIM(; L=1, seed=1)
end

@testset "RandomTFIM: rescale_disorder is the continuation step" begin
    m = RandomTFIM(; L=20, seed=3)
    @test rescale_disorder(m, 1).J == m.J
    @test all(≈(1.0), rescale_disorder(m, 0).J)        # α = 0 is the clean chain
    @test all(≈(1.0), rescale_disorder(m, 0).h)
    @test rescale_disorder(m, 0.5).J ≈ sqrt.(m.J)
    # The realisation is kept: the couplings do not change order.
    @test sortperm(rescale_disorder(m, 0.3).J) == sortperm(m.J)
    @test_throws ErrorException rescale_disorder(m, -1)
end

@testset "RandomTFIM: the MPO is the Hamiltonian it says it is" begin
    for (L, seed) in ((3, 1), (4, 2), (6, 3))
        m = RandomTFIM(; L=L, seed=seed)
        sites = siteinds("Qubit", L)
        @test dense_mpo(m, sites) ≈ dense_random_tfim(m.J, m.h) atol = 1e-12
    end
end

@testset "RandomTFIM: constant couplings reduce to TFIM" begin
    L, J, h = 6, 0.8, 1.3
    sites = siteinds("Qubit", L)
    m = RandomTFIM(fill(J, L - 1), fill(h, L))
    @test dense_mpo(m, sites) ≈ dense_mpo(TFIM(; J=J, h=h, site=SiteType("Qubit")), sites) atol =
        1e-12
end

@testset "RandomTFIM: bonds are lattice bonds" begin
    m = RandomTFIM(; L=6, seed=4)
    @test_throws ErrorException bond_term(m, 1, 3)     # not nearest neighbour
    @test_throws ErrorException bond_term(m, 6, 7)     # past the last bond
    @test_throws ErrorException onsite_term(m, 7)
end
