module MRFingerprintingReconCUDAExt

using MRFingerprintingRecon, MRFingerprintingRecon.MRICoilSensitivities
using MRFingerprintingRecon.LinearAlgebra, MRFingerprintingRecon.LinearOperators
using MRFingerprintingRecon.NonuniformFFTs, MRFingerprintingRecon.FFTW
using MRFingerprintingRecon.IterativeSolvers

using CUDA

include("NFFTNormalOpBasisFunc.jl")
include("BackProjection.jl")
include("CoilMaps.jl")

end # module