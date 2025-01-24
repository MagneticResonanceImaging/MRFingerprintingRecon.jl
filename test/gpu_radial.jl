using Pkg
Pkg.activate(".")
Pkg.instantiate()

##

using CUDA
using LinearAlgebra
using MRFingerprintingRecon
using BenchmarkTools
using Test
using FFTW
using IterativeSolvers 
using ImagePhantoms
using NFFT
using Random

Random.seed!(42)

## set parameters
T  = Float32
Nx = 128
Nc = 4
Nt = 80
Ncyc = 10

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
[x[i,:,1] .*= exp( 1im * π * i/Nx) for i ∈ axes(x,1)]
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
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up trajectory
α_g = 2π / (1+√5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi/2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

delay = (3.0/Nx, 1.0/Nx, 0.0)
trj = kooshball(2Nx, theta, phi; delay)
trj = [trj[i][1:2,:] for i ∈ eachindex(trj)]

## set up basis functions
U = randn(T, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}}(undef, size(trj[1], 2), Nt, Ncoil)
data = [data[:,it,:] for it = 1:Nt]

nfftplan = plan_nfft(trj[1], (Nx,Nx))
xcoil = copy(x)
for icoil ∈ 1:Ncoil
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data,1)
        nodes!(nfftplan, trj[it])
        xt = reshape(reshape(xcoil, :, Nc) * U[it,:], Nx, Nx)
        @views mul!(data[it][:,icoil], nfftplan, xt)
    end
end

# create mask with false value to remove
it_rm = 1
icyc_rm = 5
maskDataSelection = trues(2Nx, Ncyc, Nt)
maskDataSelection[:, icyc_rm, it_rm] .= false

maskDataSelection = reshape(maskDataSelection,:,Nt)
maskDataSelection = [vec(maskDataSelection[:,i]) for i in axes(maskDataSelection,2)]

# remove data
for it ∈ 1:Nt
    data[it] = data[it][maskDataSelection[it],:] 
    trj[it] = trj[it][:,maskDataSelection[it]] 
end

img_shape = (Nx,Nx)

# Write to CUDA arrays
data_d  = [CuArray(data[i])  for i ∈ eachindex(data)]
trj_d   = [CuArray(trj[i])   for i ∈ eachindex(trj)]
U_d     =  CuArray(U)
cmaps_d = [CuArray(cmaps[i]) for i ∈ eachindex(cmaps)]

## CPU
A = NFFTNormalOp(img_shape, trj, U; cmaps=cmaps)
b = calculateBackProjection(data, trj, cmaps; U)
xr = cg(A, vec(b), maxiter=50)
xr = reshape(xr, img_shape..., Nc)

## GPU
A_d = NFFTNormalOp(img_shape, trj_d, U_d; cmaps=cmaps_d)
b_d = calculateBackProjection(data_d, trj_d, cmaps_d; U=U_d)
xr_d = cg(A_d, vec(b_d), maxiter=50)
xr_d = reshape(Array(xr_d), img_shape..., Nc)

## Test equivalence CPU and GPU code
@test xr ≈ xr_d rtol=1e-3

## Time both
bm_cpu = @benchmark            cg(A,   vec(b),   maxiter=50)
bm_gpu = @benchmark CUDA.@sync cg(A_d, vec(b_d), maxiter=50)
t_cpu  = minimum(bm_cpu).time
t_gpu  = minimum(bm_gpu).time

@test t_gpu < t_cpu/5
