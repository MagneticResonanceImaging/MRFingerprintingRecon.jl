using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using Test

##
T  = Float32
Nx = 32
Nr = 2Nx
Nt = 1
Ncoil = 9

## Create trajectory
trj = MRFingerprintingRecon.traj_cartesian(Nx, Nx, 1, Nt; T=Float32) # float for plan_nfft()
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)] # only 2D traj here

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

## Create the equivalent integer trajectory
trj = MRFingerprintingRecon.traj_cartesian(Nx, Nx, 1, Nt; T=Int32)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)] # only 2D traj here
reco = calculateBackProjection(data, trj, cmaps; U)
reco = dropdims(reco, dims=3)

@test x ≈ reco atol = 3e-5
