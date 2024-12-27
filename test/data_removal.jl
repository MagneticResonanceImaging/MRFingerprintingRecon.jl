using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NFFT
using Test

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

## Test BackProjection 
img_shape = (Nx,Nx)
b_temp = calculateBackProjection(data, trj, img_shape)

b = calculateBackProjection(data, trj, cmaps; U=U)

## construct forward operator
A = NFFTNormalOp((Nx,Nx), trj, U, cmaps=cmaps)

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
xr = cg(A, vec(b), maxiter=20)
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
