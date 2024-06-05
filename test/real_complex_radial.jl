using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using Test
using Revise

using Random
Random.seed!(42)

## set parameters
T  = Float32
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
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]


## set up trajectory
α_g = 2π / (1+√5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi/2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

trj = kooshball(2Nx, theta, phi)
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

## BackProjection
b = calculateBackProjection(data, trj, cmaps; U=U)


## construct forward operator
A = NFFTNormalOp((Nx,Nx), trj, U, cmaps=cmaps)

## reconstruct
# @benchmark xr = cg(A, vec(b), maxiter=20)
xr = cg(A, vec(b), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

# WITH MODIFICATION
# BenchmarkTools.Trial: 18 samples with 1 evaluation.
#  Range (min … max):  282.101 ms … 296.074 ms  ┊ GC (min … max): 0.64% … 0.61%
#  Time  (median):     290.512 ms               ┊ GC (median):    0.63%
#  Time  (mean ± σ):   289.712 ms ±   4.050 ms  ┊ GC (mean ± σ):  0.63% ± 0.09%

#   ▁    ▁ ▁        ▁    ▁  ▁ ▁     ▁ ▁    ▁█    ▁▁    █ ▁      ▁  
#   █▁▁▁▁█▁█▁▁▁▁▁▁▁▁█▁▁▁▁█▁▁█▁█▁▁▁▁▁█▁█▁▁▁▁██▁▁▁▁██▁▁▁▁█▁█▁▁▁▁▁▁█ ▁
#   282 ms           Histogram: frequency by time          296 ms <

#  Memory estimate: 86.21 MiB, allocs estimate: 5616818.


# ORIGINAL
# BenchmarkTools.Trial: 48 samples with 1 evaluation.
#  Range (min … max):   89.846 ms … 230.929 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):      95.045 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   105.753 ms ±  31.666 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#    █▄                                                            
#   ████▄▃▁▁▁▁▁▃▁▁▃▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▃▁▁▁▁▁▁▁▃ ▁
#   89.8 ms          Histogram: frequency by time          231 ms <

#  Memory estimate: 593.92 KiB, allocs estimate: 5138.


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


# Real U

U_real = real.(U)

## BackProjection
b2 = calculateBackProjection(data, trj, cmaps; U=U_real)

## construct forward operator
A2 = NFFTNormalOp((Nx,Nx), trj, U_real, cmaps=cmaps)

## reconstruct
# @benchmark xr2 = cg(A2, vec(b2), maxiter=20)

## reconstruct
xr2 = cg(A2, vec(b2), maxiter=20)
xr2 = reshape(xr2, Nx, Nx, Nc)

@test xr2 ≈ xr rtol = 1e-1




##
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide
heatmap(abs.(cat(reshape(xr2, Nx, :), reshape(xr, Nx, :), dims=1)), clim=(0.75, 1.25), size=(1200,600))
# heatmap(angle.(cat(reshape(xr, Nx, :), reshape(xc, Nx, :), dims=1)), size=(1200,600))