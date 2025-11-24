module MRFingerprintingReconCUDAExt

using MRFingerprintingRecon
using MRFingerprintingRecon.FFTW
using MRFingerprintingRecon.IterativeSolvers
using MRFingerprintingRecon.LinearAlgebra
using MRFingerprintingRecon.LinearOperators
using MRFingerprintingRecon.MRICoilSensitivities
using MRFingerprintingRecon.NonuniformFFTs

using CUDA

import MRFingerprintingRecon: NFFTNormalOp, calculateCoilMaps, calculateBackProjection, kooshball, kooshballGA

include("NFFTNormalOpBasisFunc.jl")
include("BackProjection.jl")
include("CoilMaps.jl")

end # module