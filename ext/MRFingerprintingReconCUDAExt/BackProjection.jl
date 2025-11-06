
function MRFingerprintingRecon.calculateBackProjection(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    # General helper variables
    Nt = size(U, 1)
    Ncoef = size(U, 2)
    Ncoil = size(data[1], 2)
    trj_v = reduce(hcat, trj)
    Uc = conj(U)

    # Kernel helper arrays
    trj_l = [size(trj[it], 2) for it in eachindex(trj)] # nr nodes per frame
    trj_c = CuArray([0; cumsum(trj_l[1:end-1])]) # cumulative sum, starting at 0
    trj_l = CuArray(trj_l)

    # Threads-and-blocks settings for kernel_bp!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(trj_l))
    threads_y = min(max_threads ÷ threads_x, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(trj_l), Nt) ./ threads) # samples as inner index

    # Plan NFFT
    p = NonuniformFFTs.NFFTPlan(trj_v, img_shape)
    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef, Ncoil)
    data = reduce(vcat, data)
    data_temp = CuArray{cT}(undef, sum(trj_l))

    # Perform backprojection
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data, 2)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_v; density_compensation)

            # Bottleneck: >99% of computation time spent on mul! op for full-scale BP, irrespective of kernel_bp! design
            @views mul!(xbp[img_idx, icoef, icoil], adjoint(p), data_temp)
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end


function MRFingerprintingRecon.calculateBackProjection(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, cmaps::AbstractVector{<:CuArray{cT,N}}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    # Run check on array sizes
    test_dimension(data, trj, U, cmaps)

    # General helper variables
    Nt = size(U, 1)
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    trj_v = reduce(hcat, trj)
    Uc = conj(U)

    # Kernel helper arrays
    trj_l = [size(trj[it], 2) for it in eachindex(trj)]
    trj_c = CuArray([0; cumsum(trj_l[1:end-1])])
    trj_l = CuArray(trj_l)

    # Threads-and-blocks settings for kernel_bp!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(trj_l))
    threads_y = min(max_threads ÷ threads_x, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(trj_l), Nt) ./ threads)

    # Plan NFFT
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend())
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef)
    xtmp = CuArray{cT}(undef, img_shape)
    data = reduce(vcat, data)
    data_temp = CuArray{cT}(undef, sum(trj_l))

    # Perform backprojection
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed CUDA.@sync for icoil ∈ eachindex(cmaps)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_v; density_compensation)

            # Bottleneck: >99% of computation time spent on mul! op for full-scale BP, irrespective of kernel_bp! design
            exec_type1!(xtmp, p, data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end


function MRFingerprintingRecon.calculateBackProjection(data::CuArray{cT}, trj::CuArray{T}, trj_l, cmaps::AbstractVector{<:CuArray{cT,N}}; U, density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    # Run check on array sizes
    # test_dimension(data, trj, U, cmaps)


    # General helper variables
    Nt = size(U, 1)
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])
    trj_v = trj
    Uc = conj(U)

    # Kernel helper arrays
    # trj_l = [size(trj[it], 2) for it in eachindex(trj)]
    trj_c = CuArray([0; cumsum(trj_l[1:end-1])])
    # trj_l = CuArray(trj_l)

    # Threads-and-blocks settings for kernel_bp!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(trj_l))
    threads_y = min(max_threads ÷ threads_x, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(trj_l), Nt) ./ threads)

    # Plan NFFT
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true, backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200))
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))

    img_idx = CartesianIndices(img_shape)
    xbp = CUDA.zeros(cT, img_shape..., Ncoef)
    xtmp = CuArray{cT}(undef, img_shape)
    # data = reduce(vcat, data)
    data_temp = CuArray{cT}(undef, sum(trj_l))

    # # Perform backprojection
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U, 2)
        for icoil ∈ eachindex(cmaps)
            @cuda threads = threads blocks = blocks kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)
            MRFingerprintingRecon.applyDensityCompensation!(data_temp, trj_v; density_compensation)

            # Bottleneck: >99% of computation time spent on mul! op for full-scale BP, irrespective of kernel_bp! design
            exec_type1!(xtmp, p, data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
    end
    return xbp
end

function MRFingerprintingRecon.calculateCoilwiseCG(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
    Ncoil = size(data[1], 2)

    AᴴA = NFFTNormalOp(img_shape, trj, U[:, 1]; verbose)
    xbp = calculateBackProjection(data, trj, img_shape; U=U[:, 1], verbose)
    x = CUDA.zeros(cT, img_shape..., Ncoil)

    for icoil = 1:Ncoil
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter, verbose, reltol=0)
    end
    return x
end

function kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)

    # ik_sub ≡ sample index within time frame it
    ik_sub = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # it ≡ current time index
    it = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Multiply data by basis elements
    if it <= Nt
        if ik_sub <= trj_l[it]
            ik = trj_c[it] + ik_sub # absolute sample index
            data_temp[ik] = data[ik, icoil] * Uc[it, icoef]
            return
        end
    end
end
