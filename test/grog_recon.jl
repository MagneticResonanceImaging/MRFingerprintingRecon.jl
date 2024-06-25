using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using Test

## set parameters
T  = Float32
Nx = 64
Nr = 2Nx
Nc = 4
Nt = 100
Ncyc = 200
Ncoil = 9

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
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

## set up trajectory
α_g = 2π / (1+√5)
phi = Float32.(α_g * (0:Nt*Ncyc-1))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi/2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = kooshball(Nr, theta, phi)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = [Matrix{Complex{T}}(undef, size(trj[1], 2), Ncoil) for _ ∈ 1:Nt]
nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ eachindex(data)
        nodes!(nfftplan, trj[it])
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views mul!(data[it][:,icoil], nfftplan, xt)
    end
end

## Ground truth reconstruction by cropping k-space
xc = fftshift(fft(x, 1:2), 1:2)
for i ∈ CartesianIndices(xc)
    if (i[1] - Nx/2)^2 + (i[2] - Nx/2)^2 > (Nx/2)^2
        xc[i] = 0
    end
end
xc = ifft(ifftshift(xc, 1:2), 1:2)

## Remove some data
data[2] = data[2][1:end-Nr,:]
trj[2]  =  trj[2][:,1:end-Nr]

## NFFT Reconstruction
xbp_rad = calculateBackProjection(data, trj, cmaps; U=U)
A_rad = NFFTNormalOp((Nx,Nx), trj, U; cmaps=cmaps)
xr = cg(A_rad, vec(xbp_rad), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)


## GROG Reconstruction
trj = radial_grog!(data, trj, Nr, (Nx,Nx))
xbp_grog = calculateBackProjection(data, trj, cmaps; U)
A_grog = FFTNormalOp((Nx,Nx), trj, U; cmaps)
xg = cg(A_grog, vec(xbp_grog), maxiter=20)
xg = reshape(xg, Nx, Nx, Nc)

## Fix irrelevant phase slope
[xg[i,j,:] .*= -exp(1im * π * (i + j - 2)/Nx) for i = 1:Nx, j = 1:Nx]

## test recon equivalence
@test xc ≈ xr  rtol = 5e-2
@test xc ≈ xg  rtol = 5e-2

##
# using Plots
# heatmap(abs.(cat(reshape(xc, Nx, :), reshape(xr, Nx, :), reshape(xg, Nx, :), dims=1)), clim=(0.75, 1.25), size=(1100,750))
# heatmap(angle.(cat(reshape(xc, Nx, :), reshape(xr, Nx, :), reshape(xg, Nx, :); dims=1)), clim=(-0.1, 1.1), size=(1100,750))
# heatmap(angle.(reshape(xr, Nx, :)) .- angle.(reshape(xg, Nx, :)), clim=(-0.05, 0.05), size=(1100,250), c=:bluesreds)