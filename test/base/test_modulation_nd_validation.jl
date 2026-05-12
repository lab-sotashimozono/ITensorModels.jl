# Regression tests for the input-validation guards added in PR #7
# (validation hardening). These catch silent wrong-answer bugs flagged
# by the /review-pr aggregated report — C1 (AxisProduct + scalar
# profile), C2 (R <= 0 / N <= 0), I2 (axis <= 0), I3 (non-finite
# ExplicitCenter), I4 (CosineRamp edge).

using ITensorModels
using LatticeCore
using LatticeCore: position, num_sites
using Lattice2D
using Test

# ---------------------------------------------------------------------
# Distance: axis must be >= 1
# ---------------------------------------------------------------------

@testset "AxialDistance / PerpendicularDistance: axis >= 1" begin
    @test_throws ErrorException AxialDistance(0)
    @test_throws ErrorException AxialDistance(-1)
    @test_throws ErrorException PerpendicularDistance(0)
    @test_throws ErrorException PerpendicularDistance(-3)
    @test AxialDistance(1).axis == 1
    @test PerpendicularDistance(2).axis == 2
end

# ---------------------------------------------------------------------
# Profile radii: positive scalar or positive iterable
# ---------------------------------------------------------------------

@testset "SinSquareProfile: R > 0" begin
    @test_throws ErrorException SinSquareProfile(0.0)
    @test_throws ErrorException SinSquareProfile(-2.5)
    @test_throws ErrorException SinSquareProfile([0.0, 4.0])
    @test_throws ErrorException SinSquareProfile([-1.0, 4.0])
    @test SinSquareProfile(3.0).R == 3.0
    @test SinSquareProfile([2.0, 4.0]).R == [2.0, 4.0]
end

@testset "SinPowerProfile: N >= 1 and R > 0" begin
    @test_throws ErrorException SinPowerProfile{0}(3.0)
    @test_throws ErrorException SinPowerProfile{-1}(3.0)
    @test_throws ErrorException SinPowerProfile{2}(0.0)
    @test_throws ErrorException SinPowerProfile{2}(-1.0)
    @test SinPowerProfile{4}(3.0).R == 3.0
    @test SinPowerProfile{6}([2.0, 4.0]).R == [2.0, 4.0]
end

@testset "CosineRampProfile: R > 0 and 0 < edge <= R" begin
    @test_throws ErrorException CosineRampProfile(0.0, 1.0)
    @test_throws ErrorException CosineRampProfile(-3.0, 1.0)
    @test_throws ErrorException CosineRampProfile(5.0, 0.0)
    @test_throws ErrorException CosineRampProfile(5.0, -1.0)
    @test_throws ErrorException CosineRampProfile(5.0, 10.0)  # edge > R
    p = CosineRampProfile(10.0, 3.0)
    @test p.R == 10.0
    @test p.edge == 3.0
end

# ---------------------------------------------------------------------
# RadialEnvelope: distance / profile-radius compatibility
# ---------------------------------------------------------------------

@testset "RadialEnvelope: scalar distance requires scalar R" begin
    # EuclideanDistance + vector R is the trap that would otherwise
    # silently produce wrong scalar weights via `p.R[d]` indexing.
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile([2.0, 4.0])
    )
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), AxialDistance(1), SinPowerProfile{4}([2.0, 4.0])
    )
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), PerpendicularDistance(2), SinSquareProfile([2.0, 4.0])
    )
end

@testset "RadialEnvelope: AxisProductDistance requires per-axis R" begin
    # AxisProduct + scalar R caught at construction (C1): without this
    # guard, evaluation crashes with a raw MethodError deep inside
    # profile_value.
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), AxisProductDistance(), SinSquareProfile(3.0)
    )
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), AxisProductDistance(), SinPowerProfile{4}(3.0)
    )
end

@testset "RadialEnvelope: CosineRampProfile + AxisProductDistance unsupported" begin
    @test_throws ErrorException RadialEnvelope(
        BoundingBoxCenter(), AxisProductDistance(), CosineRampProfile(5.0, 1.0)
    )
end

@testset "RadialEnvelope: happy-path constructions still work" begin
    env_sphere = RadialEnvelope(
        BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(3.0)
    )
    env_axial = RadialEnvelope(
        BoundingBoxCenter(), AxialDistance(1), SinPowerProfile{4}(3.0)
    )
    env_rect = RadialEnvelope(
        BoundingBoxCenter(), AxisProductDistance(), SinSquareProfile([3.0, 4.0])
    )
    env_ramp = RadialEnvelope(
        BoundingBoxCenter(), EuclideanDistance(), CosineRampProfile(5.0, 1.0)
    )
    @test env_sphere isa RadialEnvelope
    @test env_axial isa RadialEnvelope
    @test env_rect isa RadialEnvelope
    @test env_ramp isa RadialEnvelope
end

# ---------------------------------------------------------------------
# ExplicitCenter: non-finite entries rejected
# ---------------------------------------------------------------------

@testset "ExplicitCenter: rejects NaN / Inf entries" begin
    lat = Lattice2D.honeycomb(4, 4; boundary=OpenAxis())
    @test_throws ErrorException ITensorModels.center_position(
        ExplicitCenter([NaN, 0.0]), lat
    )
    @test_throws ErrorException ITensorModels.center_position(
        ExplicitCenter([Inf, 0.0]), lat
    )
    @test_throws ErrorException ITensorModels.center_position(
        ExplicitCenter([-Inf, 0.0]), lat
    )
    rc = ITensorModels.center_position(ExplicitCenter([1.5, 2.5]), lat)
    @test rc[1] ≈ 1.5
    @test rc[2] ≈ 2.5
end

# ---------------------------------------------------------------------
# PerpendicularDistance via the lattice-bound path (filling I9 gap)
# ---------------------------------------------------------------------

@testset "site_weight: PerpendicularDistance gives a tube envelope on honeycomb" begin
    # On a 2D lattice PerpendicularDistance(1) reduces to |Δr[2]|, so
    # sites sharing the same y-coordinate must share the same weight.
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    ps = collect(LatticeCore.positions(lat))
    Ry = maximum(
        abs(p[2] - ITensorModels.center_position(BoundingBoxCenter(), lat)[2]) for p in ps
    )
    env = RadialEnvelope(
        BoundingBoxCenter(), PerpendicularDistance(1), SinSquareProfile(Ry)
    )
    by_y = Dict{Float64,Vector{Int}}()
    for k in 1:num_sites(lat)
        push!(get!(by_y, ps[k][2], Int[]), k)
    end
    same_y = [v for v in values(by_y) if length(v) > 1]
    @test !isempty(same_y)
    for group in same_y
        ws = [ITensorModels.site_weight(env, lat, k) for k in group]
        @test all(w -> w ≈ ws[1], ws)
    end
end

# ---------------------------------------------------------------------
# GeometricCenter via the lattice-bound path (filling I9 gap)
# ---------------------------------------------------------------------

@testset "GeometricCenter: arithmetic mean of positions" begin
    lat = Lattice2D.honeycomb(6, 6; boundary=OpenAxis())
    ps = collect(LatticeCore.positions(lat))
    expected = sum(ps) / length(ps)
    rc = ITensorModels.center_position(GeometricCenter(), lat)
    for d in 1:length(rc)
        @test rc[d] ≈ expected[d]
    end
end
