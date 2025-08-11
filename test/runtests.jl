using MRFingerprintingRecon
using Test

# @testset "Coil Maps" begin
#     include("cmaps.jl")
# end

@testset "Recon Radial" begin
    include("reconstruct_radial.jl")
    include("reconstruct_radial_asym.jl")
end

@testset "Recon Cartesian" begin
    include("backprojection_cart.jl")
    include("reconstruct_cart_mask.jl")
    include("reconstruct_cart_trj.jl")
end

@testset "Recon data removal" begin
    include("data_removal.jl")
end

@testset "GROG" begin
    include("grog_spoke_shift.jl")
    include("grog_precalc_shift.jl")
    include("grog_recon.jl")
end