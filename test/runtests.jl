using MRFingerprintingRecon
using Test

# @testset "Coil Maps" begin
#     include("cmaps.jl")
# end

@testset "Reconstruct" begin
    @testset "Radial" begin
        include("reconstruct.jl")
    end
    @testset "Cartesian" begin
        include("reconstruct_cart.jl")
    end
end