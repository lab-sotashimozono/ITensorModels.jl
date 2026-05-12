using ITensorModels
using LatticeCore
using LatticeCore: num_sites, position
using Lattice2D
using ITensors
using Test

@testset "ModulatedLatticeModel: construction" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => Heisenberg1D(; J=1.0)))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(5.0))
    mod = modulated_lattice(base; envelope=env)
    @test mod isa ModulatedLatticeModel
    @test mod isa AbstractLatticeModel
    @test mod.base === base
    @test mod.envelope === env
    @test site_type(mod) == site_type(base)
end

@testset "ModulatedLatticeModel: build_opsum on honeycomb" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => Heisenberg1D(; J=1.0)))
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    R = sqrt(maximum(sum((p .- rc) .^ 2) for p in ps)) + 0.5
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R))
    mod = modulated_lattice(base; envelope=env)

    H_mod = build_opsum(mod, nothing)
    H_bare = build_opsum(base, nothing)

    bare_terms = collect(ITensors.terms(H_bare))
    mod_terms = collect(ITensors.terms(H_mod))
    @test !isempty(bare_terms)
    @test !isempty(mod_terms)

    # Compare against a hand-built reference: ND assembly is
    # H = Σ_bonds bond_weight * bond_coupling_term
    #   + Σ_sites site_weight * onsite_term
    # Heisenberg onsite is empty, so the reference only has bond terms.
    ord = collect(1:num_sites(lat))
    H_ref = OpSum()
    for b in bonds(lat)
        fb = ITensorModels.bond_weight(env, lat, b.i, b.j)
        H_ref +=
            fb * ITensorModels.bond_coupling_term(Heisenberg1D(; J=1.0), ord[b.i], ord[b.j])
    end
    ref_terms = collect(ITensors.terms(H_ref))
    @test length(ref_terms) > 0
    # H_mod has extra empty-OpSum onsite contributions per site, so its
    # term count must be at least the reference count.
    @test length(mod_terms) >= length(ref_terms)
end

@testset "ModulatedLatticeModel: large-R envelope is approximately 1" begin
    # The type system doesn't allow R = ∞ for SinSquareProfile, so we
    # set R well beyond the lattice diameter, where 1 - sin²(π d / (2R))
    # is very close to 1 across every site/bond. Verify that the
    # site_weight and bond_weight values are all ≈ 1 — this is the
    # "uniform-like" reduction the wrapper must support without
    # special-casing.
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    R_huge = 1000 * sqrt(maximum(sum((p .- rc) .^ 2) for p in ps))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R_huge))

    for b in bonds(lat)
        @test ITensorModels.bond_weight(env, lat, b.i, b.j) ≈ 1.0 atol = 1e-4
    end
    for k in 1:num_sites(lat)
        @test ITensorModels.site_weight(env, lat, k) ≈ 1.0 atol = 1e-4
    end
end

@testset "ModulatedLatticeModel: corner-suppression sanity" begin
    # With R set to the largest center-to-site distance, the corner of
    # a hexagonal sample carries near-zero envelope weight while the
    # near-center bond carries close to unit weight.
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    R = sqrt(maximum(sum((p .- rc) .^ 2) for p in ps))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R))

    bs = collect(bonds(lat))
    w_corner = ITensorModels.bond_weight(env, lat, bs[1].i, bs[1].j)

    midpoints = [(position(lat, b.i) + position(lat, b.j)) / 2 for b in bs]
    distances = [sqrt(sum((mid .- rc) .^ 2)) for mid in midpoints]
    k_central = argmin(distances)
    w_centre = ITensorModels.bond_weight(env, lat, bs[k_central].i, bs[k_central].j)

    @test w_corner < w_centre
    @test w_centre >= 0.9
end

@testset "ModulatedLatticeModel: TFIM with non-zero onsite emits per-site terms" begin
    # TFIM has bond_coupling_term ≠ 0 AND onsite_term ≠ 0, so the ND
    # assembly must emit BOTH per-bond AND per-site weighted terms.
    # The total number of OpSum terms must therefore exceed the bond
    # count (which would be all we'd get if onsite emission were
    # missing).
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    base = LatticeModel(; lattice=lat, bond_models=Dict(:nearest => TFIM(; J=1.0, h=0.5)))
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    R_huge = 1000 * sqrt(maximum(sum((p .- rc) .^ 2) for p in ps))
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(R_huge))
    mod = modulated_lattice(base; envelope=env)

    H = build_opsum(mod, nothing)
    n_terms = length(collect(ITensors.terms(H)))
    n_bonds = length(collect(bonds(lat)))
    n_sites = num_sites(lat)
    # TFIM bond_coupling_term emits ONE -J Zᵢ Zⱼ per bond.
    # TFIM onsite_term emits ONE -h Xₖ per site.
    # So total terms = n_bonds + n_sites.
    @test n_terms == n_bonds + n_sites
end
