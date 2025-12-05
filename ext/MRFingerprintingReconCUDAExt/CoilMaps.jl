function MRFingerprintingRecon.calculate_coil_maps(
    data::CuArray{Complex{T}},
    trj::CuArray{T},
    img_shape::NTuple{N,Int};
    U=CUDA.ones(T, size(trj)[end]),
    mask=CUDA.ones(Bool, size(trj)[2:end]),
    kernel_size=ntuple(_ -> 6, N),
    calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)), 
    eigThresh_1=0.01,
    eigThresh_2=0.9,
    nmaps=1,
    Niter_cg=100,
    verbose=false) where {N,T}

    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    calib_scale = cu(collect(img_shape ./ calib_size))
    mask_calib = reshape(all(abs.(trj) .* calib_scale .< 0.5; dims=1), size(trj, 2), :)
    mask_calib .&= mask # update mask to only take calib region of k-space in CoilwiseCG
    trj_calib = trj .* calib_scale # scale trj for correct FOV

    x = MRFingerprintingRecon.reconstruct_coilwise(data, trj_calib, calib_size; U, mask=mask_calib, Niter_cg)
    
    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(Array(kbp), img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cu(cmaps[CartesianIndices(img_shape), ic, in]) for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

# wrapper for 4D data arrays with readout lines in separate axis
function MRFingerprintingRecon.calculate_coil_maps(
    data::CuArray{Tc,4},
    trj::CuArray{T,4},
    img_shape::NTuple{N,Int};
    mask=CUDA.ones(Bool, size(trj)[2:end]),
    kwargs...) where {N,T,Tc<:Complex}

    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return calculate_coil_maps(data, trj, img_shape; mask, kwargs...)
end


## ##########################################################################
# Internal helper functions
#############################################################################
function MRFingerprintingRecon.reconstruct_coilwise(
    data::CuArray{Tc,3},
    trj::CuArray{T,3},
    img_shape;
    U=CUDA.ones(T, size(trj)[end]),
    mask=CUDA.ones(Bool, size(trj)[2:end]),
    Niter_cg=100,
    verbose=false) where {T <: Real,Tc <: Complex{T}}

    AᴴA = MRFingerprintingRecon.NFFTNormalOp(img_shape, trj, U[:, 1]; mask=mask, verbose)
    xbp = MRFingerprintingRecon.calculate_backprojection(data, trj, img_shape; U=U[:, 1], mask=mask, verbose)
    
    Ncoil = size(data, 3)
    x = CUDA.zeros(Tc, img_shape..., Ncoil)

    for icoil = axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter=Niter_cg, verbose, reltol=0)
    end
    return x
end