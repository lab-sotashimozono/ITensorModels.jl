using ITensorModels
using LatticeCore
using LatticeCore: num_sites, position
using Lattice2D
using Test

@testset "spherical_ssd: inscribed envelope vanishes at far corner" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    env_in = spherical_ssd(lat; radius=:inscribed)
    @test env_in isa RadialEnvelope
    @test env_in.center isa BoundingBoxCenter
    @test env_in.distance isa EuclideanDistance
    @test env_in.profile isa SinSquareProfile
    # The site at the maximum-distance corner lies outside the inscribed
    # disk (its Euclidean distance to the center exceeds min(extent_d) / 2);
    # site_weight clips to 0 there.
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    k_corner = argmax([sqrt(sum((p .- rc) .^ 2)) for p in ps])
    @test ITensorModels.site_weight(env_in, lat, k_corner) ≈ 0.0 atol = 1e-12
end

@testset "spherical_ssd: circumscribed envelope is non-zero except at far corner" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    env_out = spherical_ssd(lat; radius=:circumscribed)
    @test env_out isa RadialEnvelope
    @test env_out.profile isa SinSquareProfile
    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    distances = [sqrt(sum((p .- rc) .^ 2)) for p in ps]
    d_max = maximum(distances)
    for k in 1:num_sites(lat)
        w = ITensorModels.site_weight(env_out, lat, k)
        if isapprox(distances[k], d_max; atol=1e-12)
            @test w ≈ 0.0 atol = 1e-12
        else
            @test w > 0.0
        end
    end
end

@testset "spherical_ssd: N controls plateau width" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    env_n2 = spherical_ssd(lat; radius=:circumscribed, N=2)
    env_n4 = spherical_ssd(lat; radius=:circumscribed, N=4)
    @test env_n2.profile isa SinSquareProfile
    @test env_n4.profile isa SinPowerProfile{4}
    # SinPower{4} has a wider plateau than SinSquare on (0, R), so the
    # weight at any non-boundary distance must satisfy w_n4 >= w_n2.
    for k in 1:num_sites(lat)
        w2 = ITensorModels.site_weight(env_n2, lat, k)
        w4 = ITensorModels.site_weight(env_n4, lat, k)
        @test w4 >= w2 - 1e-12
    end
end

@testset "spherical_ssd: radius keyword validation" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    @test_throws ErrorException spherical_ssd(lat; radius=:bogus)
end

@testset "cylindrical_ssd: uniform perpendicular to axis" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    env = cylindrical_ssd(lat; axis=1)
    @test env isa RadialEnvelope
    @test env.distance isa AxialDistance
    @test env.distance.axis == 1

    ps = collect(LatticeCore.positions(lat))
    by_x = Dict{Float64,Vector{Int}}()
    for k in 1:num_sites(lat)
        x = ps[k][1]
        push!(get!(by_x, x, Int[]), k)
    end
    same_x_groups = [v for v in values(by_x) if length(v) > 1]
    @test !isempty(same_x_groups)
    for group in same_x_groups
        ws = [ITensorModels.site_weight(env, lat, k) for k in group]
        @test all(w -> w ≈ ws[1], ws)
    end
end

@testset "cylindrical_ssd: axis validation" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())   # 2D lattice
    @test_throws ErrorException cylindrical_ssd(lat; axis=3)
    @test_throws ErrorException cylindrical_ssd(lat; axis=0)
end

@testset "rectangular_ssd: axis-product structure" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    env = rectangular_ssd(lat)
    @test env isa RadialEnvelope
    @test env.distance isa AxisProductDistance
    @test env.profile isa SinSquareProfile
    # The profile carries an indexable per-axis radius.
    @test length(env.profile.R) == 2

    ps = collect(LatticeCore.positions(lat))
    rc = ITensorModels.center_position(BoundingBoxCenter(), lat)
    k_central = argmin([sqrt(sum((p .- rc) .^ 2)) for p in ps])
    @test 0.0 <= ITensorModels.site_weight(env, lat, k_central) <= 1.0
    @test ITensorModels.site_weight(env, lat, k_central) >= 0.85
end

@testset "rectangular_ssd: matches direct RadialEnvelope construction" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    env_factory = rectangular_ssd(lat)
    ps = collect(LatticeCore.positions(lat))
    lo_axes = reduce((a, b) -> min.(a, b), ps)
    hi_axes = reduce((a, b) -> max.(a, b), ps)
    half = (hi_axes - lo_axes) / 2
    env_direct = RadialEnvelope(
        BoundingBoxCenter(), AxisProductDistance(), SinSquareProfile(half)
    )
    for k in 1:num_sites(lat)
        @test ITensorModels.site_weight(env_factory, lat, k) ≈
            ITensorModels.site_weight(env_direct, lat, k)
    end
end
