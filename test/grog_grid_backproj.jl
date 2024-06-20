using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using SplitApplyCombine
using Test

##
T  = Float32
Nx = 32
Nr = 2Nx
Nt = 1
Ncoil = 9

## Create trajectory
trj = MRFingerprintingRecon.traj_2d_cartesian(Nx, Nx, 1, Nt; samplingRate_units=false) # unitless for plan_nfft()
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)] # only 2D traj here


## Create phantom geometry
x = shepp_logan(Nx)
# heatmap(abs.(x), clim=(0.85, 1.25))


## Simulate coil sensitivity maps
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
cmaps[:,:,1] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,2] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,3] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷8,  7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,4] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,5] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,6] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,  7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,7] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, Nx÷8),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,8] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, Nx÷2),  (Nx÷1.5,Nx÷1.5))], 2)
cmaps[:,:,9] .= phantom(1:Nx, 1:Nx, [gauss2((7Nx÷8, 7Nx÷8), (Nx÷1.5,Nx÷1.5))], 2)

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end

cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]


## Simulate data
data = [Matrix{Complex{T}}(undef, size(trj[1], 2), Ncoil) for _ ∈ 1:Nt]
nfftplan = plan_nfft(trj[1], (Nx,Nx))

xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]

    for it ∈ eachindex(data)
        nodes!(nfftplan, trj[it])
        @views mul!(data[it][:,icoil], nfftplan, xcoil)
    end
end

## Reconstruction and test
U = ones(ComplexF32, length(data), 1)

#### Convert traj to units of sampling rate
#### Required for calculateBackProjection_gridded function
for it ∈ eachindex(trj)
    @views mul!(trj[it][1,:], trj[it][1,:], Nx)
    @views mul!(trj[it][2,:], trj[it][2,:], Nx)
    trj[it] = ceil.(Int, trj[it])
end

reco = calculateBackProjection_gridded(data, trj, U, cmaps)
reco = dropdims(reco, dims=3)
@test abs.(x) ≈ abs.(reco) atol = 3e-5

# @test angle.(x) ≈ angle.(reco) atol = 3e-5 # FIXME: Phase effects?!

##
# heatmap(abs.(cat(x, reco, dims=1)), clim=(0.85, 1.25))
# heatmap(angle.(cat(x, reco, dims=1)))
