using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Plots
using Random
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

T = Float32
Nx = 32
Nc = 4 # nr of coefficients in the temporal subspace
Nt = 20 # nr of acquired time frames per cycle
Ncyc = 10 # nr of cycles (i.e., repeats of flip angle pattern)
img_shape = (Nx, Nx) # 2D image in this example

# create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:, :, 1] = transpose(shepp_logan(Nx))
x[1:end÷2, :, 1] .*= exp(1im * π / 3)
x[:, :, 2] = shepp_logan(Nx)

# coil maps
Ncoil = 9
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
[cmaps[i, :, 2] .*= exp(1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 3] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 4] .*= exp(1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 5] .*= exp(-1im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[i, :, 6] .*= exp(2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[i, :, 7] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 1)]
[cmaps[:, i, 8] .*= exp(2im * π * i / Nx) for i ∈ axes(cmaps, 2)]
[cmaps[:, i, 9] .*= exp(-2im * π * i / Nx) for i ∈ axes(cmaps, 2)]

# coil maps are defined as vectors of complex n-dimensional arrays
for i ∈ CartesianIndices(@view cmaps[:, :, 1])
    cmaps[i, :] ./= norm(cmaps[i, :])
end
cmaps = [cmaps[:, :, ic] for ic = 1:Ncoil]
nothing #hide

# set up trajectory
α_g = 2π / (1 + √5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi / 2)
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

# generate trj as (2, Nx*Ncyc, Nt)-shaped array
# alternatively, trj can be shaped as (ndims, n_ADC, n_readouts, n_timeframes)
trj = traj_kooshball(2Nx, theta, phi) # shaped (ndims, nsamples, n_timeframes)
trj = trj[1:2, :, :] # use only the first 2 dims as a 2D k-space trj

# set up basis functions
U = randn(Complex{T}, Nt, Nc)
U, _, _ = svd(U)
nothing #hide

# simulate data as (2Nx*Ncyc, Nt, Ncoil)-shaped array
# alternatively, data can be shaped as (2Nx, Ncyc, Nt, Ncoil)
data = Array{Complex{T},3}(undef, 2Nx * Ncyc, Nt, Ncoil)
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
xcoil = copy(x)

for icoil ∈ axes(data, 3)
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:, :, it], 2, :)))
        xt = reshape(reshape(xcoil, :, Nc) * U[it, :], Nx, Nx)
        # simulate data from image using type-2 (uniform to non-uniform) NUFFT
        @views NonuniformFFTs.exec_type2!(data[:, it, icoil], nfftplan, xt)
    end
end

# create sampling mask
it_rm = 1
icyc_rm = 5
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, icyc_rm, it_rm] .= false
sample_mask = reshape(sample_mask, 2Nx*Ncyc, Nt) # masks are 2- or 3-dim depending on format of trj and data
nothing #hide

cmaps = calculate_coil_maps(data, trj, img_shape; U)
nothing #hide

AᴴA = NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask)
b = calculate_backprojection(data, trj, cmaps; U, sample_mask)
nothing #hide

# solve inverse problem with CG
xr = cg(AᴴA, vec(b), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc) # reshape vector back to 2D image with Nc coefficients
nothing #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
