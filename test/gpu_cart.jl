using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
using BenchmarkTools
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using GPUArrays
using CUDA
using JLArrays
using Revise

using MRFingerprintingRecon

using Test

arrayType = CuArray

T  = Float32
Nx = 64
Nc = 4
Nt = 100
Ncoil = 9

# Simulate data on CPU

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
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

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data and backprojection
D = rand(Nx, Nx, Nt) .< 0.1
data = Array{Complex{T}}(undef, Nx, Nx, Nt)

xbp = similar(x) .= 0
# Scalar Indexing slow, but here for testing fine
for icoil = 1:Ncoil
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        data[i,:] .= U * x[i,:] .* cmaps2[icoil][i]
    end
    data .= ifftshift(data, (1,2))
    fft!(data, (1,2))
    data .*= ifftshift(D, (1,2))
    ifft!(data, (1,2))
    data .= fftshift(data, (1,2))
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        xbp[i,:] .+= U' * data[i,:] * conj(cmaps2[icoil][i])
    end
end

## Move to device

D_device = arrayType{eltype(D)}(undef, (size(D)))
@allowscalar copyto!(D_device, D)

U_device = arrayType{eltype(U)}(undef, (size(U)))
@allowscalar copyto!(U_device, U)

xbp_device = arrayType{eltype(xbp)}(undef, (size(xbp)))
@allowscalar copyto!(xbp_device, xbp)

cmaps_device = arrayType{eltype(cmaps)}(undef, (size(cmaps)))
@allowscalar copyto!(cmaps_device, cmaps)
cmaps_device = [cmaps_device[:,:,ic] for ic=1:Ncoil]


## GPU Reconstruction
A_device = FFTNormalOp(D_device, U_device; cmaps=cmaps_device)
xr_device = arrayType(zeros(Complex{T}, size(vec(x))))
xbp_device = reshape(xbp_device, Nx*Nx*Nc) # Linearize, while preserving arraytype
@benchmark CUDA.@sync cg!(xr_device, A_device, xbp_device, maxiter=20)
cg!(xr_device, A_device, xbp_device, maxiter=20)
xr_device = reshape(xr_device, Nx, Nx, Nc)


## CPU Reconstruction
A = FFTNormalOp(D, U; cmaps=cmaps2)
xr = Array(zeros(Complex{T}, size(vec(x))))
xbp = reshape(xbp, Nx*Nx*Nc) # Linearize, while preserving arraytype
@benchmark cg!(xr, A, xbp, maxiter=20)
cg!(xr, A, xbp, maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)


## test equivalence of CPU and GPU
gpu = Array{eltype(xr_device)}(undef, (size(xr_device)))
copyto!(gpu, xr_device)

cpu = Array{eltype(xr)}(undef, (size(xr)))
copyto!(cpu, xr)

mask = abs.(x[:,:,1]) .> 0
@test gpu[mask,:] ≈ cpu[mask,:] rtol = 1e-3
@test gpu[mask,:] ≈ x[mask,:] rtol = 1e-3


## Visualization
# using Plots
# plotly()
# heatmap(abs.(reshape(xr2, Nx, :)), clim=(0.75, 1.25))
# heatmap(abs.(cat(reshape(cpu, Nx, :), reshape(gpu, Nx, :), reshape(x, Nx, :), dims=1)), clim=(0.75, 1.25))