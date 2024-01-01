using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using SplitApplyCombine
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

# [cmaps[i,:,2] .*= exp( 1im * π/4 * i/Nx) for i ∈ axes(cmaps,1)]
# [cmaps[i,:,3] .*= exp(-1im * π/4 * i/Nx) for i ∈ axes(cmaps,1)]
# [cmaps[:,i,4] .*= exp( 1im * π/4 * i/Nx) for i ∈ axes(cmaps,2)]
# [cmaps[:,i,5] .*= exp(-1im * π/4 * i/Nx) for i ∈ axes(cmaps,2)]
# [cmaps[i,:,6] .*= exp( 2im * π/4 * i/Nx) for i ∈ axes(cmaps,1)]
# [cmaps[i,:,7] .*= exp(-2im * π/4 * i/Nx) for i ∈ axes(cmaps,1)]
# [cmaps[:,i,8] .*= exp( 2im * π/4 * i/Nx) for i ∈ axes(cmaps,2)]
# [cmaps[:,i,9] .*= exp(-2im * π/4 * i/Nx) for i ∈ axes(cmaps,2)]

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up trajectory
α_g = 2π / (1+√5)
phi = α_g * (0:Nt*Ncyc-1)
theta = 0 * (1:Nt*Ncyc) .+ pi/2
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = kooshball(Nr, theta, phi; T = T)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, Ncoil)
nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data,2)
        nodes!(nfftplan, trj[it])
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views mul!(data[:,it,icoil], nfftplan, xt)
    end
end

## self-calibrating radial GROG
lnG = scGROG(reshape(data, Nr, :, Ncoil), trj)

## GROG Reconstruction
xbp_grog, Λ, D = griddedBackProjection(reshape(copy(data), Nr, :, Ncoil), lnG, deepcopy(trj), U, cmaps; density=true, verbose=true)

A_grog_efficient = FFTNormalOpBasisFuncLO((Nx,Nx), U; cmaps=cmaps, Λ=Λ, verbose=true)
xg = cg(A_grog_efficient, vec(xbp_grog), maxiter=20)
xg = reshape(xg, Nx, Nx, Nc)

A_grog_default = FFTNormalOpBasisFuncLO((Nx,Nx), U; cmaps=cmaps, D=D, verbose=true)
xgd = cg(A_grog_default, vec(xbp_grog), maxiter=20)
xgd = reshape(xgd, Nx, Nx, Nc)

## Fix irrelevant phase slope
[xg[i,j,:]  .*= -exp(1im * π * (i + j)/Nx) for i = 1:Nx, j = 1:Nx]
[xgd[i,j,:] .*= -exp(1im * π * (i + j)/Nx) for i = 1:Nx, j = 1:Nx]

## NFFT Reconstruction
xbp_rad = calculateBackProjection(data, trj, U, cmaps)
A_rad = NFFTNormalOpBasisFuncLO((Nx,Nx), trj, U; cmaps=cmaps)
xr = cg(A_rad, vec(xbp_rad), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## crop x
xc = fftshift(fft(x, 1:2), 1:2)
for i ∈ CartesianIndices(xc)
    if (i[1] - Nx/2)^2 + (i[2] - Nx/2)^2 > (Nx/2)^2
        xc[i] = 0
    end
end
xc = ifft(ifftshift(xc, 1:2), 1:2)

## test recon equivalence
@test xc ≈ xr  rtol = 1e-1
@test xc ≈ xg  rtol = 2e-1
@test xc ≈ xgd rtol = 2e-1
@test xg ≈ xgd rtol = 1e-2

## test equivalence of efficient kernel calculation
@test A_grog_default.prod!.A.Λ ≈ A_grog_efficient.prod!.A.Λ

## test GROG kernels for 1st spoke in trajectory
data = reshape(data, Nr, :, Ncoil)
trjr = reshape(combinedimsview(trj), 2, Nr, :)

for ispoke in rand(axes(data,2), 20) # test 20 random spokes
    nm = dropdims(diff(trjr[:,1:2,ispoke],dims=2),dims=2) .* Nr
    @test [data[j,ispoke,:][ic] for j in 2:Nr, ic = 1:Ncoil] ≈ [(exp(nm[1] * lnG[1]) * exp(nm[2] * lnG[2]) * data[j,ispoke,:])[ic] for j in 1:Nr-1, ic =1:Ncoil] rtol = 3e-1
end

##
# using Plots
# heatmap(abs.(cat(reshape(xc, Nx, :), reshape(xr, Nx, :), reshape(xg, Nx, :), reshape(xgd, Nx, :), dims=1)), clim=(0.75, 1.25), size=(1100,800))
# heatmap(angle.(cat(reshape(xc, Nx, :), reshape(xr, Nx, :), reshape(xg, Nx, :), reshape(xgd, Nx, :), dims=1)), clim=(0.75, 1.25), size=(1100,800))