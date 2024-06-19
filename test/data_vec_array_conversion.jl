using BenchmarkTools
using MRFingerprintingRecon
using SplitApplyCombine
using Test

using Random
Random.seed!(42)

## set parameters
Nx = 32
Nr = 2Nx
Nt = 10
Ncyc = 20
Ncoil = 9
Nrep = 4

## Case 1: Non-Cartesian data, 3 Dims, joined cycles and samples

data = randn(Nr*Ncyc, Nt, Ncoil)
ref = copy(data)

data = MRFingerprintingRecon.data_vec(data)
data = MRFingerprintingRecon.data_array(data)

@test data ≈ ref  rtol = 1e-7


## Case 2: Non-Cartesian data, 4 Dims

data = randn(Nr*Ncyc, Nt, Ncoil, Nrep)
ref = copy(data)

data = MRFingerprintingRecon.data_vec(data)
data = MRFingerprintingRecon.data_array(data)

@test data ≈ ref  rtol = 1e-7


## Case 3: Cartesian data, 3 Dims

data = randn(Nx, Nx, Nrep)
ref = copy(data)

data = MRFingerprintingRecon.data_vec(data; Cartesian=true)
data = MRFingerprintingRecon.data_array(data; Cartesian=true)

@test data ≈ ref  rtol = 1e-7