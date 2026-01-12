using MRISubspaceRecon
using Test

@testset "Recon Radial" begin
    include("cmaps.jl")
    include("reconstruct_radial.jl")
    include("reconstruct_radial_asym.jl")
    include("data_removal.jl")
end

@testset "Recon Cartesian" begin
    include("backprojection_cart.jl")
    include("reconstruct_cart_mask.jl")
    include("reconstruct_cart_trj.jl")
end

@testset "GROG" begin
    include("grog_spoke_shift.jl")
    include("grog_precalc_shift.jl")
    include("grog_recon.jl")
end

@testset "Wrapper" begin
    include("wrapper.jl")
end