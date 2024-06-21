using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using SplitApplyCombine
using Test

using Random
Random.seed!(42)

##
T  = Float32
Nx = 32
Nr = 2Nx
Nt = 100
Ncoil = 9

## Create trajectory
trj = MRFingerprintingRecon.traj_2d_radial_goldenratio(Nr, 1, Nt; N=1)

## Create phantom geometry
x = shepp_logan(Nx)


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
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, Ncoil)
nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data,2)
        nodes!(nfftplan, trj[it])
        @views mul!(data[:,it,icoil], nfftplan, xcoil)
    end
end

# Rearrange data into vector
data = [data[:,i,:] for i=1:size(data,2)]

## Test GROG kernels for some spokes in golden ratio based trajectory

lnG = MRFingerprintingRecon.grog_calib(data, trj, Nr)

for ispoke ∈ rand(axes(data, 1), 42) # test randomly 42 spokes

    # Distance matrix θn
    nm = dropdims(diff(trj[ispoke][:,1:2],dims=2),dims=2) .* Nr

    # d1 = [data[j,ispoke,:][ic] for j in 1:Nr-1, ic = 1:Ncoil]

    d1_shifted = [(exp(nm[1] * lnG[1]) * exp(nm[2] * lnG[2]) * data[ispoke][j,:])[ic] for j in 1:Nr-1, ic =1:Ncoil]

    d2 = [data[ispoke][j,:][ic] for j in 2:Nr, ic = 1:Ncoil]

    # FIXME: Relative tolerance the best choice?
    @test d1_shifted ≈ d2 rtol = 3e-1
end

