using MRFingerprintingRecon
using Test

# @testset "Coil Maps" begin
#     include("cmaps.jl")
# end

@testset "Recon Radial" begin
    include("reconstruct_radial.jl")
end

@testset "Recon Cartesian" begin
    include("backprojection_cart.jl")
    include("reconstruct_cart.jl")
end

@testset "Recon data removal" begin
    include("data_removal.jl")
end

@testset "GROG" begin
    include("grog_spoke_shift.jl")
    include("grog_precalc_shift.jl")
    include("grog_recon.jl")
end


# FIXME: Find way to incorporate GPU tests into CI pipeline
# @testset "GPU Tests" begin
#     include("gpu_cart.jl")
# end