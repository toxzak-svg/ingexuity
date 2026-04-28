#!/usr/bin/env julia
# Run tests for IngExuity
using Test
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using IngExuity

@testset "IngExuity Test Suite" begin
    include(joinpath(@__DIR__, "modules.jl"))
end