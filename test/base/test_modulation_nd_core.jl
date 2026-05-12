using ITensorModels
using Test

@testset "RadialEnvelope: type construction" begin
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(3.0))
    @test env isa RadialEnvelope
    @test env isa AbstractModulationND
    @test env isa AbstractModulation
    @test env.center isa BoundingBoxCenter
    @test env.distance isa EuclideanDistance
    @test env.profile isa SinSquareProfile
end

@testset "Center constructors" begin
    @test GeometricCenter() isa AbstractCenter
    @test BoundingBoxCenter() isa AbstractCenter
    @test ExplicitCenter([1.0, 2.0]) isa AbstractCenter
    @test ExplicitCenter((1.0, 2.0)).r == (1.0, 2.0)
end

@testset "Distance constructors" begin
    @test EuclideanDistance() isa AbstractDistance
    @test AxialDistance(1) isa AbstractDistance
    @test AxialDistance(1).axis == 1
    @test PerpendicularDistance(2).axis == 2
    @test AxisProductDistance() isa AbstractDistance
end

@testset "distance_at_position: Euclidean" begin
    @test distance_at_position(EuclideanDistance(), [0.0, 0.0], [0.0, 0.0]) ≈ 0.0
    @test distance_at_position(EuclideanDistance(), [3.0, 4.0], [0.0, 0.0]) ≈ 5.0
    @test distance_at_position(EuclideanDistance(), [1.0, 1.0], [-1.0, -1.0]) ≈ 2sqrt(2)
    @test distance_at_position(EuclideanDistance(), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) ≈ 1.0
end

@testset "distance_at_position: Axial" begin
    @test distance_at_position(AxialDistance(1), [3.0, 4.0], [0.0, 0.0]) ≈ 3.0
    @test distance_at_position(AxialDistance(2), [3.0, 4.0], [0.0, 0.0]) ≈ 4.0
    @test distance_at_position(AxialDistance(2), [3.0, 4.0], [0.0, 6.0]) ≈ 2.0
end

@testset "distance_at_position: Perpendicular" begin
    # In 2D, perpendicular distance to axis 1 is |Δr[2]|.
    @test distance_at_position(PerpendicularDistance(1), [3.0, 4.0], [0.0, 0.0]) ≈ 4.0
    # In 3D, perpendicular to axis 3 is sqrt(Δr[1]^2 + Δr[2]^2).
    @test distance_at_position(PerpendicularDistance(3), [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]) ≈
        5.0
    @test distance_at_position(PerpendicularDistance(2), [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]) ≈
        sqrt(3.0^2 + 5.0^2)
end

@testset "distance_at_position: AxisProduct" begin
    ds = distance_at_position(AxisProductDistance(), [3.0, 4.0], [0.0, 0.0])
    @test ds == (3.0, 4.0)
    ds2 = distance_at_position(AxisProductDistance(), [3.0, -4.0, 5.0], [1.0, 2.0, 0.0])
    @test ds2 == (2.0, 6.0, 5.0)
end

@testset "profile_value: SinSquareProfile" begin
    p = SinSquareProfile(4.0)
    @test profile_value(p, 0.0) ≈ 1.0
    @test profile_value(p, 4.0) ≈ 0.0
    @test profile_value(p, 8.0) ≈ 0.0    # outside R clipped to 0
    # cos²(π/8) at d = R/2 = 2
    @test profile_value(p, 2.0) ≈ cos(pi / 4)^2 ≈ 0.5
end

@testset "profile_value: SinPowerProfile{N}" begin
    p2 = SinPowerProfile{2}(4.0)
    p4 = SinPowerProfile{4}(4.0)
    p8 = SinPowerProfile{8}(4.0)
    pss = SinSquareProfile(4.0)
    # N=2 ≡ SinSquare.
    for d in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
        @test profile_value(p2, d) ≈ profile_value(pss, d)
    end
    # Boundary and center.
    @test profile_value(p4, 0.0) ≈ 1.0
    @test profile_value(p4, 4.0) ≈ 0.0
    @test profile_value(p4, 5.0) ≈ 0.0    # outside R clipped
    # Larger N => wider central plateau, i.e. value(p4, d) >= value(p2, d) on (0, R).
    for d in [0.5, 1.0, 1.5, 2.0]
        @test profile_value(p4, d) >= profile_value(p2, d)
        @test profile_value(p8, d) >= profile_value(p4, d)
    end
end

@testset "profile_value: CosineRampProfile" begin
    p = CosineRampProfile(10.0, 3.0)
    # Plateau region [0, R - edge] = [0, 7].
    @test profile_value(p, 0.0) ≈ 1.0
    @test profile_value(p, 5.0) ≈ 1.0
    @test profile_value(p, 7.0) ≈ 1.0
    # Boundary.
    @test profile_value(p, 10.0) ≈ 0.0
    @test profile_value(p, 15.0) ≈ 0.0
    # Midpoint of ramp (d = 8.5, halfway through the ramp): value = 0.5.
    @test profile_value(p, 8.5) ≈ 0.5
end

@testset "profile_value: AxisProduct path" begin
    p = SinSquareProfile([4.0, 6.0])
    @test profile_value(p, (0.0, 0.0)) ≈ 1.0
    @test profile_value(p, (4.0, 0.0)) ≈ 0.0
    @test profile_value(p, (0.0, 6.0)) ≈ 0.0
    # Product of per-axis envelopes.
    expected = cos(pi * 2.0 / (2 * 4.0))^2 * cos(pi * 3.0 / (2 * 6.0))^2
    @test profile_value(p, (2.0, 3.0)) ≈ expected
end

@testset "profile_value: SinPowerProfile AxisProduct (N=2 ≡ SinSquare)" begin
    p2 = SinPowerProfile{2}([4.0, 6.0])
    pss = SinSquareProfile([4.0, 6.0])
    for ds in [(0.0, 0.0), (1.0, 1.0), (2.0, 3.0), (3.5, 5.5)]
        @test profile_value(p2, ds) ≈ profile_value(pss, ds)
    end
end

@testset "profile_value: CosineRamp + AxisProduct unsupported" begin
    p = CosineRampProfile(5.0, 1.0)
    @test_throws ErrorException profile_value(p, (1.0, 2.0))
end
