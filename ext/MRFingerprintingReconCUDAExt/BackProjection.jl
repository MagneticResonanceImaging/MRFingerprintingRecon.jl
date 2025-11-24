function MRFingerprintingRecon.calculateBackProjection(data::CuArray{Tc,3}, trj::CuArray{T,3}, img_shape; U=cu(I(size(trj)[end])), mask=CUDA.ones(Bool, size(trj)[2:end]), density_compensation=:none, verbose=false) where {T<:Real,Tc<:Complex{T}}
    nsamp_t = dropdims(sum(mask; dims=1); dims=1) # number of samples per time frame
    Ncoef = size(U, 2)
    Ncoil = size(data, 3)
    Uc = conj(U)

    # Compute the last index per time frame, and insert zero at position 1
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t)) 
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1]) # cumulative sum indicates to which time frame a sample belongs

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    trj_rs = trj[:, mask] # select subset of trj
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs)) # transform matrix to tuples, change sign of FT exponent, change range to (0,2π)

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(Tc, img_shape..., Ncoef, Ncoil)
    data_temp = CuArray{Tc}(undef, sum(nsamp_t))

    # Apply sampling mask to data and perform backprojection
    data_rs = data[mask, :]
    threads, blocks = default_kernel_config(nsamp_t)
    verbose && (CUDA.synchronize(); println("calculating backprojection..."); flush(stdout))
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data, 3)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data_rs, Uc, nsamp_t, cumsum_nsamp, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_rs; density_compensation)
            @views exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp) # type 1: non-uniform points to uniform grid
        end
        verbose && (CUDA.synchronize(); println("coefficient = $icoef: t = $t s"); flush(stdout))
    end
    return xbp
end

function MRFingerprintingRecon.calculateBackProjection(data::CuArray{Tc,3}, trj::CuArray{T,3}, cmaps::AbstractVector{<:CuArray{Tc,N}}; U=cu(I(size(trj)[end])), mask=CUDA.ones(Bool, size(trj)[2:end]), density_compensation=:none, verbose=false) where {T<:Real,Tc<:Complex{T},N}
    nsamp_t = dropdims(sum(mask; dims=1); dims=1) # number of samples per time frame
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    Uc = conj(U)
    
    # Compute the last index per time frame, and insert zero at position 1
    cumsum_nsamp = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))
    cumsum_nsamp[2:end] = cumsum(nsamp_t[1:end-1]) # cumulative sum indicates to which time frame a sample belongs

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    trj_rs = trj[:, mask]
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(Tc, img_shape..., Ncoef)
    xtmp = CuArray{Tc}(undef, img_shape)
    data_temp = CuArray{Tc}(undef, sum(nsamp_t))

    # Apply sampling mask to data and perform backprojection
    data_rs = data[mask, :]
    threads, blocks = default_kernel_config(nsamp_t)
    verbose && (CUDA.synchronize(); println("calculating backprojection..."); flush(stdout))
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data_rs, Uc, nsamp_t, cumsum_nsamp, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_rs; density_compensation)
            exec_type1!(xtmp, p, data_temp) # type 1: non-uniform points to uniform grid
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && (CUDA.synchronize(); println("coefficient = $icoef: t = $t s"); flush(stdout))
    end
    return xbp
end

# Wrapper for use with 4D arrays, where nr of ADC samples per readout is in a separate at 2ⁿᵈ dim
function MRFingerprintingRecon.calculateBackProjection(data::CuArray{Tc,4}, trj::CuArray{T,4}, arg3; mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...) where {T,Tc<:Complex}
    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return MRFingerprintingRecon.calculateBackProjection(data, trj, arg3; kwargs..., mask)
end

function calculateCoilwiseCG(data::CuArray{Tc,3}, trj::CuArray{T,3}, img_shape; U=CUDA.ones(T, size(trj)[end]), mask=CUDA.ones(Bool, size(trj)[2:end]), maxiter=100, verbose=false) where {T<:Real,Tc<:Complex{T}}
    Ncoil = size(data, 3)

    AᴴA = MRFingerprintingRecon.NFFTNormalOp(img_shape, trj, U[:, 1]; mask=mask, verbose)
    xbp = MRFingerprintingRecon.calculateBackProjection(data, trj, img_shape; U=U[:, 1], mask=mask, verbose)
    x = CUDA.zeros(Tc, img_shape..., Ncoil)

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

    # Multiply data by basis, accounting for varying number of samples per time frame
    if it <= length(nsamp_t)
        if ik <= nsamp_t[it]
            ik_abs = cumsum_nsamp[it] + ik # absolute sample index
            data_temp[ik_abs] = data[ik_abs, icoil] * Uc[it, icoef]
            return
        end
    end
end

# Default threading for backprojection
function default_kernel_config(nsamp_t)
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, length(nsamp_t))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(nsamp_t), length(nsamp_t)) ./ threads) # samples as inner index
    return threads, blocks
end