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
Nc = 4
Nt = 100
Ncyc = 100

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = shepp_logan(Nx)
x[1:end÷2,:,1] .*= exp(1im * π/3)

## set up trajectory
α_g = 2π / (1+√5)
phi = α_g * (1:Nt*Ncyc)
theta = 0 * (1:Nt*Ncyc) .+ pi/2
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = kooshball(2Nx, theta, phi; T = T)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, 1)
xr = reshape(x, :, Nc)
nfftplan = plan_nfft(trj[1], (Nx,Nx))
for it ∈ axes(data,2)
    nodes!(nfftplan, trj[it])
    xt = reshape(xr * U[it,:], Nx, Nx)
    @views mul!(data[:,it,1], nfftplan, xt)
end

## BackProjection
b = vec(calculateBackProjection(data, trj, U, [ones(T, Nx,Nx)], verbose = true))

## construct forward operator
A = NFFTNormalOpBasisFuncLO((Nx,Nx), trj, U; verbose = true)

## test forward operator
λ = zeros(Complex{T}, Nc, Nc, 2Nx*2Nx)
for i ∈ eachindex(A.prod!.A.kmask_indcs)
    λ[:,:,A.prod!.A.kmask_indcs[i]] .= A.prod!.A.Λ[:,:,i]
end
λ = reshape(λ, Nc, Nc, 2Nx, 2Nx)

for i = 1:Nc, j = 1:Nc
    l1 = conj.(λ[i,j,:,:])
    l2 = λ[j,i,:,:]
    l2 = conj.(fft(conj.(ifft(l2))))
    @test l1 ≈ l2 rtol = 1e-4
end

## reconstruct
xr = similar(b)
xr .= 0
cg!(xr, A, b, maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## crop x
xc = fftshift(fft(x, 1:2), 1:2)
for i ∈ CartesianIndices(xc)
    if (i[1] - Nx/2)^2 + (i[2] - Nx/2)^2 > (Nx/2)^2
        xc[i] = 0
    end
end
xc = ifft(ifftshift(xc, 1:2), 1:2)

##
@test xr ≈ xc rtol = 1e-1

##
# using Plots
# plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide
# heatmap(abs.(cat(reshape(xr, Nx, :), reshape(xc, Nx, :), dims=1)))
# heatmap(angle.(cat(reshape(xr, Nx, :), reshape(xc, Nx, :), dims=1)))