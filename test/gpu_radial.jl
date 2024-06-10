using Pkg
Pkg.activate(".")
Pkg.instantiate()
# ] dev ~/git/NFFT.jl/CuNFFT
##
using BenchmarkTools
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using CuNFFT
using GPUArrays
using JLArrays
using CUDA
using Revise
using SplitApplyCombine

using MRFingerprintingRecon

using Test

arrayType = CuArray

## set parameters
T  = Float32
Nd = 3
Nx = 32
Nc = 4
Nt = 20
Ncyc = 10

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
Ncoil = 9
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
[cmaps[i,:,2] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,3] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,4] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,5] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[i,:,6] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,7] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,8] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,9] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,2)]

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps2 = [cmaps[:,:,ic] for ic=1:Ncoil]


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
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, Ncoil)
nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps2[icoil]
    for it ∈ axes(data,2)
        nodes!(nfftplan, trj[it])
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views mul!(data[:,it,icoil], nfftplan, xt)
    end
end

## Move to device
trj_cv = combinedimsview(trj)
trj_device = arrayType{eltype(trj_cv)}(undef, (size(trj_cv)))
@allowscalar copyto!(trj_device, trj_cv)
trj_device = [trj_device[:,:,it] for it=1:Nt]

U_device = arrayType{eltype(U)}(undef, (size(U)))
@allowscalar copyto!(U_device, U)

data_device = arrayType{eltype(data)}(undef, (size(data)))
@allowscalar copyto!(data_device, data)

cmaps_device = arrayType{eltype(cmaps)}(undef, (size(cmaps)))
@allowscalar copyto!(cmaps_device, cmaps)
cmaps_device = [cmaps_device[:,:,ic] for ic=1:Ncoil]


## Test equivalence of CPU and GPU: Backprojection

#### GPU
b_device = calculateBackProjection(data_device, trj_device, cmaps_device; U=U_device)
gpu_b = Array{eltype(b_device)}(undef, (size(b_device)))
copyto!(gpu_b, b_device)


#### CPU 
b = calculateBackProjection(data, trj, cmaps2; U=U)
cpu_b = Array{eltype(b)}(undef, (size(b)))
copyto!(cpu_b, b)

@test gpu_b ≈ cpu_b rtol = 1e-3


## Test equivalence of CPU and GPU: Toeplitz Kernel

#### GPU
Λ_device, kmask_indcs_device = MRFingerprintingRecon.calculateToeplitzKernelBasis(2 .* (Nx,Nx), trj_device, U_device)
gpu_Λ = Array{eltype(Λ_device)}(undef, (size(Λ_device)))
copyto!(gpu_Λ, Λ_device)
gpu_k_mask = Array{eltype(kmask_indcs_device)}(undef, (size(kmask_indcs_device)))
copyto!(gpu_k_mask, kmask_indcs_device)

#### CPU
Λ, kmask_indcs = MRFingerprintingRecon.calculateToeplitzKernelBasis(2 .* (Nx,Nx), trj, U)

@test gpu_Λ ≈ Λ rtol = 1e-3
@test gpu_k_mask ≈ kmask_indcs rtol = 1e-3

## Test equivalence of CPU and GPU: Reconstruction

#### GPU
b_device = reshape(b_device, Nx*Nx*Nc) # Linearize, while preserving arraytype
A_device = NFFTNormalOp((Nx,Nx), trj_device, U_device; cmaps=cmaps_device)
xr_device = cg(A_device, b_device, maxiter=20)

xr_device = reshape(xr_device, Nx, Nx, Nc)
gpu = Array{eltype(xr_device)}(undef, (size(xr_device)))
copyto!(gpu, xr_device)

#### CPU
A = NFFTNormalOp((Nx,Nx), trj, U, cmaps=cmaps2)
xr = cg(A, vec(b), maxiter=20)

xr = reshape(xr, Nx, Nx, Nc)
cpu = Array{eltype(xr)}(undef, (size(xr)))
copyto!(cpu, xr)

mask = abs.(x[:,:,1]) .> 0
@test gpu[mask,:] ≈ cpu[mask,:] rtol = 1e-3


## Benchmarks

#### Backprojections
println("GPU Backprojection")
@benchmark CUDA.@sync b_device = calculateBackProjection(data_device, trj_device, cmaps_device; U=U_device)
println("CPU Backprojection")
@benchmark b = calculateBackProjection(data, trj, cmaps2; U=U)

#### Reconstructions
println("GPU Reconstruction")
@benchmark CUDA.@sync xr_device = cg(A_device, b_device, maxiter=20)
println("CPU Reconstruction")
@benchmark xr = cg(A, vec(b), maxiter=20)

## Visualization
# using Plots
# plotly()
# heatmap(abs.(reshape(xr, Nx, :)), clim=(0.75, 1.25))
# heatmap(abs.(cat(reshape(cpu_b, Nx, :), reshape(gpu_b, Nx, :), dims=1)))
# heatmap(abs.(cat(reshape(cpu, Nx, :), reshape(gpu, Nx, :), reshape(x, Nx, :), dims=1)), clim=(0.75, 1.25))
# heatmap(abs.(cat(reshape(cpu, Nx, :), reshape(gpu, Nx, :), reshape(x, Nx, :), dims=1)))