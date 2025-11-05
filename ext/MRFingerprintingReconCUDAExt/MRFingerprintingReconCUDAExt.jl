module MRFingerprintingReconCUDAExt

using MRFingerprintingRecon, MRFingerprintingRecon.LinearAlgebra, MRFingerprintingRecon.NonuniformFFTs, MRFingerprintingRecon.FFTW, MRFingerprintingRecon.IterativeSolvers, MRFingerprintingRecon.LinearOperators
using CUDA

include("NFFTNormalOpBasisFunc.jl")
include("BackProjection.jl")

end # module