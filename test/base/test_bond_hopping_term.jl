using ITensorModels
using ITensors
using ITensorMPS
using LinearAlgebra: norm
using Test

# `bond_coupling_term` must be exactly the forward hopping, its conjugate, and whatever
# density-density piece the model carries. Pinning the DECOMPOSITION is what keeps a consumer
# that scales the forward block from drifting away from the Hamiltonian it came from.
function _decomposes(m, st, i, j; extra=nothing)
    sites = siteinds(st, j + 1)
    mpo(H) = prod(MPO(H, sites))
    K = mpo(bond_hopping_term(m, i, j))
    rebuilt = K + swapprime(dag(K), 0 => 1)
    # `MPO` of an empty `OpSum` throws, so an absent density piece is absent, not zero.
    extra === nothing || (rebuilt += mpo(extra))
    full = mpo(bond_coupling_term(m, i, j))
    return norm(full - rebuilt), norm(full)
end

@testset "bond_hopping_term: coupling = forward + conjugate (+ density)" begin
    for (m, st, extra) in (
        (RiceMele1D(; v=0.7, w=1.3, Δ=0.4), "Fermion", nothing),
        (RiceMeleHubbard1D(; v=0.7, w=1.3, Δ=0.4, U=2.0), "Electron", nothing),
    )
        for (i, j) in ((1, 2), (2, 3))
            d, n = _decomposes(m, st, i, j; extra=extra)
            @test d < 1.0e-12 * max(n, 1.0)
            @test n > 1.0e-6
        end
    end
    # With a nearest-neighbour interaction the density piece is the remainder, and naming it
    # is required: leaving it out must fail.
    m = RiceMele1D(; v=0.7, w=1.3, Δ=0.4, V=0.9)
    dens = OpSum()
    dens += (0.9, "N", 1, "N", 2)
    d_with, n = _decomposes(m, "Fermion", 1, 2; extra=dens)
    d_without, _ = _decomposes(m, "Fermion", 1, 2)
    @test d_with < 1.0e-12 * n
    @test d_without > 0.1
end

@testset "bond_hopping_term: it is FORWARD only, and carries the Peierls phase" begin
    # The whole reason it exists: the phase multiplies this block and nothing else, so a
    # driven consumer scales a cached operator rather than rebuilding the model.
    sites = siteinds("Fermion", 3)
    m0 = RiceMele1D(; v=0.7, w=1.3, Δ=0.4)
    mA = RiceMele1D(; v=0.7, w=1.3, Δ=0.4, A=0.35)
    K0 = prod(MPO(bond_hopping_term(m0, 1, 2), sites))
    KA = prod(MPO(bond_hopping_term(mA, 1, 2), sites))
    @test norm(KA - cis(-0.35 * bond_displacement(mA)) * K0) < 1.0e-12 * norm(K0)
    # Not the conjugate as well: the forward block alone is NOT Hermitian at A != 0.
    @test norm(KA - swapprime(dag(KA), 0 => 1)) > 0.1 * norm(KA)
end
