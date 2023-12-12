using MRFingerprintingRecon
using Test

# @testset "Coil Maps" begin
#     include("cmaps.jl")
# end

@testset "Recon Radial" begin
    include("reconstruct_radial.jl")
end
@testset "Recon Cartesian" begin
    include("reconstruct_cart.jl")
end
