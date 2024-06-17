module MRFingerprintingRecon

using Polyester
using LinearAlgebra
using FFTW
using NFFT
using CuNFFT
using NFFTTools
using MRICoilSensitivities
using LinearOperators
using SplitApplyCombine
using ExponentialUtilities
using GPUArrays
using JLArrays
using CUDA

export FFTNormalOp, radial_grog!, calculateBackProjection_gridded
export NFFTNormalOp, calcCoilMaps, calculateBackProjection, kooshball, kooshballGA, calcFilteredBackProjection

function __init__()
    if Threads.nthreads() > 1
      BLAS.set_num_threads(1)
    end
    FFTW.set_num_threads(Threads.nthreads())
end

include("GROG.jl")
include("FFTNormalOpBasisFunc.jl")
include("NFFTNormalOpBasisFunc.jl")
include("CoilMaps.jl")
include("BackProjection.jl")
include("Trajectories.jl")
include("deprecated.jl")

end # module
