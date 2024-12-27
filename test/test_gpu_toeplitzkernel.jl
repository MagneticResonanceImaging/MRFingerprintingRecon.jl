# Compare the Toeplitz kernel calculated with CUDA and the NonuniformsFFTs package to 
# the CPU NFFT package standard. Comparison done via numerical comparison wuth @test

using Pkg
Pkg.activate(".") 
Pkg.instantiate()

#

using NonuniformFFTs
using Serialization
using Plots
using CUDA
using LinearAlgebra
using MRFingerprintingRecon
using BenchmarkTools
using Test
using FFTW

##

function get_kernel_cart(kernel, indices, img_shape_os)
    kernel_full = zeros(eltype(kernel), (size(kernel, 1), size(kernel, 2), img_shape_os[1] * img_shape_os[2] * img_shape_os[3]))
    kernel_full[:, :, indices] = copy(kernel)
    kernel_full = reshape(kernel_full, (size(kernel, 1), size(kernel, 2), img_shape_os[1], img_shape_os[2], img_shape_os[3]))
    return kernel_full
end


## Read brain data from file
indir = "/gpfs/scratch/maatmi01/kreal_project/kreal_gpu/data/full/"
suffix = "HSFP_mid2195"

data = open("$indir/data_$suffix.jls", "r") do file
    deserialize(file)
end
cmaps = open("$indir/cmaps_$suffix.jls", "r") do file
    deserialize(file)
end
trj = open("$indir/trj_$suffix.jls", "r") do file
    deserialize(file)
end
U = open("$indir/U_$suffix.jls", "r") do file
    deserialize(file)
end

# Shorten data for speed
# Reduce to 50 time points and img_shape=(256,256,192): 171 s
# Reduce to 50 time points and img_shape=(128,128, 96):  25 s (4 coeff)
# Reduce to 50 time points and img_shape=(128,128, 96):  13 s (4 coeff)

# data = data[1:50]
# trj = trj[1:50]
U = U[:, 1:4]

# Write to CUDA arrays
data_d  = [CuArray(data[i])  for i ∈ eachindex(data)]
trj_d   = [CuArray(trj[i])   for i ∈ eachindex(trj)]
U_d     =  CuArray(U)
cmaps_d = [CuArray(cmaps[i]) for i ∈ eachindex(cmaps)]

##

img_shape = (256, 256, 192)
img_shape_os = 2 .* img_shape

## kmask cpu
kmask_cpu = MRFingerprintingRecon.calculate_kmask_indcs(img_shape_os, trj)

## kmask gpu: difference with cpu of 10 pixels at 1 k_z step
kmask_gpu = MRFingerprintingRecon.calculate_kmask_indcs(img_shape_os, trj_d)

## CPU kernel
Λ_cpu, kmask_indcs_cpu = MRFingerprintingRecon.calculateToeplitzKernelBasis(img_shape_os, trj, U; verbose=true)

## GPU kernel
Λ_gpu, kmask_indcs_gpu = MRFingerprintingRecon.calculateToeplitzKernelBasis(img_shape_os, trj_d, U_d; verbose=true)
Λ_gpu = Array(Λ_gpu)
kmask_indcs_gpu = Array(kmask_indcs_gpu)

## Account for very slightly different kmask_indcs
Λ_cpu_full = get_kernel_cart(Λ_cpu, kmask_indcs_cpu, img_shape_os)
Λ_gpu_full = get_kernel_cart(Λ_gpu, kmask_indcs_gpu, img_shape_os)

##

@test Λ_cpu_full ≈ Λ_gpu_full rtol=1e-3

##

A = NFFTNormalOp(img_shape, trj, U)