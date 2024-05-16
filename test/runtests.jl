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

@testset "GROG" begin
    include("grog_spoke_shift.jl")
    include("grog_grid_backproj.jl")
    include("grog_precalc_shift.jl")
    include("grog_recon.jl")
end

# @testset "GPU Tests" begin
#     include("gpu_cart.jl")
# end