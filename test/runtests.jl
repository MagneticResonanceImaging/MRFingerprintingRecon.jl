using MRFingerprintingRecon
using Test

# @testset "Coil Maps" begin
#     include("cmaps.jl")
# end

@testset "Reconstruct" begin
    include("reconstruct.jl")
    include("reconstruct_cart.jl")
end