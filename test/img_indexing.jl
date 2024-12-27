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

x = rand(128, 128)