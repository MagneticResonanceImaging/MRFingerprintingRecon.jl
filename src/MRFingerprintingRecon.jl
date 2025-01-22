module MRFingerprintingRecon

using Polyester
using LinearAlgebra
using FFTW
using MRICoilSensitivities
using LinearOperators
using ExponentialUtilities
using IterativeSolvers

# Additional packages for GPU functionality
using CUDA
using NonuniformFFTs # Solves CUDA NFFT crash when number of samples >2.2 mill

export NFFTNormalOp, calcCoilMaps, calculateBackProjection, kooshball, kooshballGA, calcFilteredBackProjection
export FFTNormalOp, radial_grog!

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
