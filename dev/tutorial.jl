#md # [![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/USER/REPO/main?filepath=docs/src/build_literate/tutorial.ipynb)

# # Non-Cartesian MRI

# This example illustrates how to perform an MRI subspace reconstruction from non-Cartesian k-space data.
# For this example, we need the following packages:
using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using NonuniformFFTs
using Plots
using Random
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
# We first simulate some data from a Shepp-Logan phantom and generate some coil maps using
# various phase modulations:
T = Float32
Nx = 32
Nc = 4 # nr of coefficients in the temporal subspace
Nt = 20 # nr of acquired time frames per cycle
Ncyc = 10 # nr of cycles (i.e., repeats of flip angle pattern)
img_shape = (Nx, Nx) # 2D image in this example

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:, :, 1] = transpose(shepp_logan(Nx))
x[1:end÷2, :, 1] .*= exp(1im * π / 3)
x[:, :, 2] = shepp_logan(Nx)

## coil maps
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

## coil maps are defined as vectors of complex n-dimensional arrays
for i ∈ CartesianIndices(@view cmaps[:, :, 1])
    cmaps[i, :] ./= norm(cmaps[i, :])
end
cmaps = [cmaps[:, :, ic] for ic = 1:Ncoil]
nothing #hide #!nb
# Next, we set up a kooshball trajectory for data acquisition and generate a set of basis functions. The non-Cartesian methods
# use float trajectories in range $k \in [-0.5, 0.5)$, as opposed to integer trajectories for Cartesian methods.
## set up trajectory
α_g = 2π / (1 + √5)
phi = Float32.(α_g * (1:Nt*Ncyc))
theta = Float32.(0 * (1:Nt*Ncyc) .+ pi / 2) 
phi = reshape(phi, Ncyc, Nt)
theta = reshape(theta, Ncyc, Nt)

## generate trj as (2, Nx*Ncyc, Nt)-shaped array
## alternatively, trj can be shaped as (ndims, n_ADC, n_readouts, n_timeframes)
trj = traj_kooshball(2Nx, theta, phi) # shaped (ndims, nsamples, n_timeframes)
trj = trj[1:2, :, :] # use only the first 2 dims as a 2D k-space trj

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U, _, _ = svd(U)
nothing #hide #!nb
# We use the phantom image `x`, the coil maps `cmaps`, the trajectory `trj`, and the basis functions `U` to simulate some k-space data:
## simulate data as (2Nx*Ncyc, Nt, Ncoil)-shaped array
## alternatively, data can be shaped as (2Nx, Ncyc, Nt, Ncoil)
data = Array{Complex{T},3}(undef, 2Nx * Ncyc, Nt, Ncoil)
nfftplan = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
xcoil = copy(x)

for icoil ∈ axes(data, 3)
    xcoil .= x
    xcoil .*= cmaps[icoil]
    for it ∈ axes(data, 2)
        set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reshape(trj[:, :, it], 2, :)))
        xt = reshape(reshape(xcoil, :, Nc) * U[it, :], Nx, Nx)
        ## simulate data from image using type-2 (uniform to non-uniform) NUFFT
        @views NonuniformFFTs.exec_type2!(data[:, it, icoil], nfftplan, xt)
    end
end
# The data format uses either 3D or 4D arrays, where the 4D format is used to place ADC points within a separate array axis from the total number of samples.
# Internally, all code relies on 3D arrays and the 4D arrays are handled by wrappers.
# Furthermore, reconstructions can make use of a binary mask to exclude specific samples from being included in the reconstruction. 
# To illustrate the data removal, we create a mask that removes one time frame from one cycle:
## create sampling mask
it_rm = 1
icyc_rm = 5
sample_mask = trues(2Nx, Ncyc, Nt)
sample_mask[:, icyc_rm, it_rm] .= false
sample_mask = reshape(sample_mask, 2Nx*Ncyc, Nt) # masks are 2- or 3-dim depending on format of trj and data
nothing #hide #!nb
# Coil maps may also be auto-calibrated from k-space measurements using ESPIRiT:
cmaps = calculate_coil_maps(data, trj, img_shape; U)
nothing #hide #!nb
# Now, we can compute the normal operator and the adjoint NUFFT (backprojection) with the specified sampling mask:
AᴴA = NFFTNormalOp(img_shape, trj, U; cmaps, sample_mask)
b = calculate_backprojection(data, trj, cmaps; U, sample_mask)
nothing #hide #!nb
# GPU-methods are called through multiple dispatch, i.e., when objects of type `CuArray` are passed as arguments. The normal operator `A` and the backprojection `b` are compatible with the iterative solvers from [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and 
# [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl). This enables solving the inverse problem with various algorithms, including conjugate gradient (CG):
## solve inverse problem with CG
xr = cg(AᴴA, vec(b), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc) # reshape vector back to 2D image with Nc coefficients
nothing #hide #!nb
