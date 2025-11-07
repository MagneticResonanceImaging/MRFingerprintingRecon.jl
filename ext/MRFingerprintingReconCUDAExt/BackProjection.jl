function MRFingerprintingRecon.calculateBackProjection(data::CuArray{cT}, trj::CuArray{T}, nsamp_t, img_shape; U, density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T}}

    # General helper variables
    Nt = size(U, 1)
    Ncoef = size(U, 2)
    Ncoil = size(data, 2)
    trj_v = trj
    Uc = conj(U)

    # Kernel helper arrays
    trj_c = CuArray([0; cumsum(nsamp_t[1:end-1])])

    # Threads-and-blocks settings for kernel_bp!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(nsamp_t), Nt) ./ threads) # samples as inner index

    # Plan non-uniform FFT
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef, Ncoil)
    data_temp = CuArray{cT}(undef, sum(nsamp_t))

    # Perform backprojection
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data, 2)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, nsamp_t, trj_c, Nt, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_v; density_compensation)
            @views exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp) 
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

function MRFingerprintingRecon.calculateBackProjection(data::CuArray{cT}, trj::CuArray{T}, nsamp_t, cmaps::AbstractVector{<:CuArray{cT,N}}; U, density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}

    # General helper variables
    Nt = size(U, 1)
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    trj_v = trj
    Uc = conj(U)

    # Kernel helper arrays
    trj_c = CuArray([0; cumsum(nsamp_t[1:end-1])])

    # Threads-and-blocks settings for kernel_bp!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(nsamp_t), Nt) ./ threads)

    # Plan NFFT
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef)
    xtmp = CuArray{cT}(undef, img_shape)
    data_temp = CuArray{cT}(undef, sum(nsamp_t))

    ## Perform backprojection
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U, 2)
        for icoil ∈ eachindex(cmaps)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, nsamp_t, trj_c, Nt, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_v; density_compensation)
            exec_type1!(xtmp, p, data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
    end
    return xbp
end

function MRFingerprintingRecon.calculateCoilwiseCG(data::CuArray{cT}, trj::CuArray{T}, nsamp_t, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
    Ncoil = size(data, 2)

    AᴴA = MRFingerprintingRecon.NFFTNormalOp(img_shape, trj, nsamp_t, U[:, 1]; verbose)
    xbp = MRFingerprintingRecon.calculateBackProjection(data, trj, nsamp_t, img_shape; U=U[:, 1], verbose)
    x = CUDA.zeros(cT, img_shape..., Ncoil)

    for icoil = 1:Ncoil
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter, verbose, reltol=0)
    end
    return x
end

## ##########################################################################
# Internal use
#############################################################################
function kernel_bp!(data_temp, data, Uc, nsamp_t, trj_c, Nt, icoef, icoil)

    # ik_sub ≡ sample index within time frame it
    ik_sub = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # it ≡ current time index
    it = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Multiply data by basis elements
    if it <= Nt
        if ik_sub <= nsamp_t[it]
            ik = trj_c[it] + ik_sub # absolute sample index
            data_temp[ik] = data[ik, icoil] * Uc[it, icoef]
            return
        end
    end
end