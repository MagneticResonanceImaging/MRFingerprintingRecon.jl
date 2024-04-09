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
Nt = 100
Ncoil = 9
Nrep = 3
Nd = 2

## Create trajectory
trj = MRFingerprintingRecon.goldenratio(Nr, 1, Nt; N=1)
trj = [trj[i][1:Nd,:] for i ∈ eachindex(trj)] # only 2D traj, here

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

# Create repeating pattern
trj2 = repeat(deepcopy(trj), outer = [1, Nrep])

data2 = repeat(deepcopy(data), outer = [1, 1, 1, Nrep])


## Calibrate GROG kernel
lnG = MRFingerprintingRecon.grog_calculatekernel(data, trj, Nr)
lnG2 = MRFingerprintingRecon.grog_calculatekernel(data2, trj2, Nr)

@test lnG ≈ lnG2 rtol = 1e-6


## Gridding of each sample with non repeating trajectory (Reference)
MRFingerprintingRecon.grog_grid_only!(data, trj, lnG, Nr, (Nx,Nx))

## Exploit Precalculated Shifts
data3 = deepcopy(data2)
trj3 = deepcopy(trj2)
MRFingerprintingRecon.grog_grid_only!(data3, trj3, lnG2, Nr, (Nx,Nx))

## Gridding of each sample with individual shift estimation
data4 = deepcopy(data2)
trj4 = deepcopy(trj2)

# Join time and repetition dimension to calculate shifts individually
data4 = permutedims(data4, (1,2,4,3))
data4 = reshape(data4, Nr, :, Ncoil)
trj4 = reshape(combinedimsview(trj4), Nd, Nr, :)
trj4 = [trj4[:,:,t] for t=1:Nrep*Nt]

MRFingerprintingRecon.grog_grid_only!(data4, trj4, lnG2, Nr, (Nx,Nx))

# Undo formatting for comparison with data3 and trj3
data4 = reshape(data4, Nr, Nt, Nrep, Ncoil)
data4 = permutedims(data4, (1,2,4,3))
trj4 = reshape(combinedimsview(trj4), Nd, Nr, Nt, Nrep)
trj4 = [trj4[:,:,t,r] for t=1:Nt, r=1:Nrep]


## Compare gridding with and without repeating pattern
@test combinedimsview(trj) ≈ combinedimsview(trj3)[:,:,:,1] rtol = 1e-6
@test data ≈ data3[:,:,:,1] rtol = 1e-6

## Compare gridding of repeating pattern with and without exploitation of repeating patterns
@test trj3 ≈ trj4 rtol = 1e-6
@test data3 ≈ data4 rtol = 1e-6