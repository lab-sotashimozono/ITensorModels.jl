# Regression tests for the "extension-missing" fallback methods added
# in PR #37 (advisory A7) and refined in PR #9 (I2) to include
# typeof(lat) in the error message.
#
# In the test environment LatticeCoreExt IS loaded, so the
# AbstractLattice-typed methods win for actual lattice arguments. The
# fallbacks fire when the caller passes a non-AbstractLattice value as
# `lat`. We exercise that path by calling each fallback with a Dict
# (clearly not an AbstractLattice) and asserting the error message
# names both the function signature and the offending type.

using ITensorModels
using LatticeCore
using Test

const _BOGUS_LAT = Dict(:x => 1)

@testset "Extension-missing fallback: center_position" begin
    try
        ITensorModels.center_position(BoundingBoxCenter(), _BOGUS_LAT)
        @test false
    catch e
        @test e isa ErrorException
        @test occursin("center_position", e.msg)
        @test occursin("Dict", e.msg)
        @test occursin("LatticeCore", e.msg)
    end
end

@testset "Extension-missing fallback: distance_at" begin
    try
        ITensorModels.distance_at(EuclideanDistance(), _BOGUS_LAT, 1, [0.0, 0.0])
        @test false
    catch e
        @test e isa ErrorException
        @test occursin("distance_at", e.msg)
        @test occursin("Dict", e.msg)
        @test occursin("LatticeCore", e.msg)
    end
end

@testset "Extension-missing fallback: site_envelope" begin
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(3.0))
    try
        ITensorModels.site_envelope(env, _BOGUS_LAT, 1)
        @test false
    catch e
        @test e isa ErrorException
        @test occursin("site_envelope", e.msg)
        @test occursin("Dict", e.msg)
        @test occursin("LatticeCore", e.msg)
    end
end

@testset "Extension-missing fallback: site_weight (ND signature)" begin
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(3.0))
    try
        ITensorModels.site_weight(env, _BOGUS_LAT, 1)
        @test false
    catch e
        @test e isa ErrorException
        @test occursin("site_weight", e.msg)
        @test occursin("Dict", e.msg)
        @test occursin("LatticeCore", e.msg)
    end
end

@testset "Extension-missing fallback: bond_weight (ND signature)" begin
    env = RadialEnvelope(BoundingBoxCenter(), EuclideanDistance(), SinSquareProfile(3.0))
    try
        ITensorModels.bond_weight(env, _BOGUS_LAT, 1, 2)
        @test false
    catch e
        @test e isa ErrorException
        @test occursin("bond_weight", e.msg)
        @test occursin("Dict", e.msg)
        @test occursin("LatticeCore", e.msg)
    end
end

@testset "1D site_weight / bond_weight signatures are NOT shadowed" begin
    # The 1D (mod, i::Int, L::Int) signatures must still resolve to the
    # 1D AbstractModulation methods even though RadialEnvelope is also
    # an AbstractModulationND. The ND fallbacks live at
    # (::AbstractModulationND, lat::Any, ::Int) and
    # (::AbstractModulationND, lat::Any, ::Int, ::Int) -- different
    # second/third positional argument types from the 1D protocol so
    # dispatch is unambiguous.
    L = 60
    @test ITensorModels.site_weight(SSD(), 30, L) > 0
    @test ITensorModels.bond_weight(SSD(), 30, L) > 0
end
