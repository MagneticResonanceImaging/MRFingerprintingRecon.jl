module MRFingerprintingRecon

using Polyester
using LinearAlgebra
using MRIReco
using NFFT
using FFTW

function __init__()
    if Threads.nthreads() > 1
      BLAS.set_num_threads(1)
    end
    FFTW.set_num_threads(Threads.nthreads())
end

include("NFFTNormalOpBasisFunc.jl")

end # module
