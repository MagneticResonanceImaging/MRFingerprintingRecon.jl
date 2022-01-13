module MRFingerprintingRecon

using Polyester
using LinearAlgebra
using FFTW
using NFFT
import MRIReco

export NFFTNormalOpBasisFunc, calcCoilMaps, calculateBackProjection, kooshball, kooshballGA

function __init__()
    if Threads.nthreads() > 1
      BLAS.set_num_threads(1)
    end
    FFTW.set_num_threads(Threads.nthreads())
end

include("NFFTNormalOpBasisFunc.jl")
include("CoilMaps.jl")
include("BackProjection.jl")
include("Trajectories.jl")

end # module
