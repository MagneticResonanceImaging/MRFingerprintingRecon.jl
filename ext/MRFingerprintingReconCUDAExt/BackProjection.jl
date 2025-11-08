function MRFingerprintingRecon.calculateBackProjection(data::CuArray{cT}, trj::CuArray{T}, nsamp_t::CuArray{<:Integer}, img_shape; U, density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T}}

    Ncoef = size(U, 2)
    Ncoil = size(data, 2)
    Uc = conj(U)
    cumsum_nsamp = cumsum(nsamp_t[1:end-1]) |> x -> cat(CUDA.zeros(eltype(x), 1), x; dims=1);

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef, Ncoil)
    data_temp = CuArray{cT}(undef, sum(nsamp_t))

    threads, blocks = default_kernel_config(nsamp_t)
    verbose && (CUDA.synchronize(); println("calculating backprojection..."); flush(stdout))
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data, 2)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, nsamp_t, cumsum_nsamp, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj; density_compensation)
            @views exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp) 
        end
        verbose && (CUDA.synchronize(); println("coefficient = $icoef: t = $t s"); flush(stdout))
    end
    return xbp
end

function MRFingerprintingRecon.calculateBackProjection(data::CuArray{cT}, trj::CuArray{T}, nsamp_t::CuArray{<:Integer}, cmaps::AbstractVector{<:CuArray{cT,N}}; U, density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}

    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    Uc = conj(U)
    cumsum_nsamp = cumsum(nsamp_t[1:end-1]) |> x -> cat(CUDA.zeros(eltype(x), 1), x; dims=1);

    # Plan NFFT
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef)
    xtmp = CuArray{cT}(undef, img_shape)
    data_temp = CuArray{cT}(undef, sum(nsamp_t))

    ## Perform backprojection
    threads, blocks = default_kernel_config(nsamp_t)
    verbose && (CUDA.synchronize(); println("calculating backprojection..."); flush(stdout))
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, nsamp_t, cumsum_nsamp, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj; density_compensation)
            exec_type1!(xtmp, p, data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && (CUDA.synchronize(); println("coefficient = $icoef: t = $t s"); flush(stdout))
    end
    return xbp
end

function MRFingerprintingRecon.calculateCoilwiseCG(data::CuArray{cT}, trj::CuArray{T}, nsamp_t::CuArray{<:Integer}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
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
function kernel_bp!(data_temp, data, Uc, nsamp_t, cumsum_nsamp, icoef, icoil)
    ik = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    it = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Multiply data by basis elements
    if it <= length(nsamp_t)
        if ik <= nsamp_t[it]
            ik_abs = cumsum_nsamp[it] + ik # absolute sample index
            data_temp[ik_abs] = data[ik_abs, icoil] * Uc[it, icoef]
            return
        end
    end
end

function default_kernel_config(nsamp_t)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, length(nsamp_t))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(nsamp_t), length(nsamp_t)) ./ threads) # samples as inner index
    return threads, blocks
end