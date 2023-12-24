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
Nt = 5
Nc = 2
Ncyc = 100
Ncoil = 9

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)

## coil maps
cmaps = zeros(Complex{T}, Nx, Nx, Ncoil)
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

## set up basis functions
U = randn(Complex{T}, Nt, 2)
U,_,_ = svd(U)

## set up trajectory
α_g = 2π / (1+√5)
phi = α_g * (1:Nt*Ncyc)
theta = 0 * (1:Nt*Ncyc) .+ pi/2
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = kooshball(2Nx, theta, phi; T = T)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)]

## simulate data
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, Ncoil)
nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil[:,:,1] .*= cmaps[icoil]
    for it ∈ axes(data,2)
        nodes!(nfftplan, trj[it])
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views mul!(data[:,it,icoil], nfftplan, xt)
    end
end

## self-calibrating radial GROG
G = scGROG(reshape(data, 2Nx, :, Ncoil), trj)

## GROG Reconstruction
xbp_grog, Λ, D = griddedBackProjection(reshape(data, 2Nx, :, Ncoil), G, trj, U, cmaps; density=true)
A_grog_default = FFTNormalOpBasisFuncLO((Nx,Nx), U; cmaps=cmaps, D=D)
A_grog_efficient = FFTNormalOpBasisFuncLO((Nx,Nx), U; cmaps=cmaps, Λ=Λ)
xg = zeros(Complex{T}, size(vec(x)))
cg!(xg, A_grog_efficient, vec(xbp_grog), maxiter=20)
xg = reshape(xg, Nx, Nx, Nc)
xgd = zeros(Complex{T}, size(vec(x)))
cg!(xgd, A_grog_default, vec(xbp_grog), maxiter=20)
xgd = reshape(xgd, Nx, Nx, Nc)

## NFFT Reconstruction
xbp_rad = calculateBackProjection(data, trj, U, cmaps)
A_rad = NFFTNormalOpBasisFuncLO((Nx,Nx), trj, U; cmaps=cmaps)
xr = similar(vec(xbp_rad))
xr .= 0
cg!(xr, A_rad, vec(xbp_rad), maxiter=20)
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
mask = abs.(x[:,:,1]) .> 0
@test abs.(xc[mask,:]) ≈ abs.(xr[mask,:])  rtol = 1e-1
@test abs.(xc[mask,:]) ≈ abs.(xg[mask,:])  rtol = 5e-1
@test abs.(xc[mask,:]) ≈ abs.(xgd[mask,:]) rtol = 5e-1
@test abs.(xg[mask,:]) ≈ abs.(xgd[mask,:]) rtol = 1e-2

## test equivalence of efficient kernel calculation
for i ∈ CartesianIndices((Nx, Nx))
    @test A_grog_default.prod!.A.Λ[:,:,i] ≈ Λ[:,:,i] rtol = 2e-1
end

## test GROG kernels for 1st spoke in trajectory
data = reshape(data, 2Nx, :, Ncoil)
trjr = reshape(combinedimsview(trj), 2, 2Nx, :)
nm = dropdims(diff(trjr[:,1:2,:],dims=2),dims=2)' .* 2Nx #nyquist units
data_temp = Array{Complex{T}}(undef, Ncoil)
for j = 1:2Nx-1
    data_temp .= data[j,1,:]
    for k = 2:-1:1
        data_temp .= G[k]^nm[k] * data_temp
    end
    @test data_temp ≈ data[j+1,1,:] rtol = 5e-1
end

##
# using Plots
# plot(heatmap(abs.(cat(xc[:,:,1], xr[:,:,1], xg[:,:,1], xgd[:,:,1], dims=2)), aspect_ratio=1), size=(1000,400))