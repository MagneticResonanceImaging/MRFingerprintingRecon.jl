"""
    calculateBackProjection(data, trj, img_shape; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps::AbstractVector{<:AbstractArray{cT,N}}; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps_img_shape; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps; U)

Calculate (filtered) backprojection

# Arguments
- `data <: Union{AbstractVector{<:AbstractMatrix{cT}},AbstractMatrix{cT}}`: Complex dataset either as AbstractVector of matrices or single matrix. The optional outer matrix defines different time frames that are reconstructed in the subspace defined in U.
- `trj <: AbstractVector{<:AbstractMatrix{T}}`: Trajectory with samples corresponding to the dataset. For a Cartesian reconstruction, use `T <: Int` and define `trj[it][idim,ik] ∈ (1, img_shape[idim])`. If `T <: Float`, the NFFT is used.

One of the following arguments needs to be supplied
- `img_shape::NTuple{N,Int}`: Shape of image; in this case, the data is reconstructed coilwise.
- `cmaps::::AbstractVector{<:AbstractArray{T}}`: Coil sensitivities; in this case, the coils are added up to a single backprojection.

# Optional Keyword Arguments
- `U::Matrix = I(length(data))` or `= I(1)``: Basis coefficients of subspace (only defined if data and trj have different timeframes)
- `density_compensation = :none`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type `AbstractVector{<:AbstractVector}`
- `verbose::Boolean = false`: Verbosity level
"""

function calculateBackProjection(data::AbstractVector{<:AbstractArray{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    
    Ncoef = size(U, 2)

    trj_v = reduce(hcat, trj)
    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))

    Ncoil = size(data[1], 2)
    xbp = Array{cT}(undef, img_shape..., Ncoef, Ncoil)

    trj_idx = cumsum([size(trj[it],2) for it in eachindex(trj)])
    data_temp = Vector{cT}(undef, trj_idx[end])

    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data[1], 2)
            @simd for it in eachindex(data)
                idx1 = (it == 1) ? 1 : trj_idx[it-1] + 1
                idx2 = trj_idx[it]
                @views data_temp[idx1:idx2] .= data[it][:, icoil] .* conj(U[it, icoef])
            end
            applyDensityCompensation!(data_temp, trj_v; density_compensation)
            @views exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp)
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}

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
            @cuda threads=threads blocks=blocks kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)
            applyDensityCompensation!(data_temp, trj_v; density_compensation)

            # Bottleneck: >99% of computation time spent on mul! op for full-scale BP, irrespective of kernel_bp! design
            @views mul!(xbp[img_idx, icoef, icoil], adjoint(p), data_temp)
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractVector{<:AbstractMatrix{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, cmaps::AbstractVector{<:AbstractArray{cT,N}}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    test_dimension(data, trj, U, cmaps)

    trj_v = reduce(hcat, trj)
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_v))
    xbp = zeros(cT, img_shape..., Ncoef)
    xtmp = Array{cT}(undef, img_shape)

    trj_idx = cumsum([size(trj[it],2) for it in eachindex(trj)])
    data_temp = Vector{cT}(undef, trj_idx[end])
    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @simd for it in eachindex(data)
                idx1 = (it == 1) ? 1 : trj_idx[it-1] + 1
                idx2 = trj_idx[it]
                @views data_temp[idx1:idx2] .= data[it][:, icoil] .* conj(U[it, icoef])
            end
            applyDensityCompensation!(data_temp, trj_v; density_compensation)
            exec_type1!(xtmp, p, data_temp)
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, cmaps::AbstractVector{<:CuArray{cT, N}}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}

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
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @cuda threads=threads blocks=blocks kernel_bp!(data_temp, data, Uc, trj_l, trj_c, Nt, icoef, icoil)
            applyDensityCompensation!(data_temp, trj_v; density_compensation)

            # Bottleneck: >99% of computation time spent on mul! op for full-scale BP, irrespective of kernel_bp! design
            exec_type1!(xtmp, p, data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractArray{cT}, trj::AbstractMatrix{T}, cmaps_img_shape; density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T}}
    return calculateBackProjection([data], [trj], cmaps_img_shape; U=I(1), density_compensation, verbose)
end

# Method for GROG gridded data / trajectory
function calculateBackProjection(data::AbstractVector{<:AbstractArray}, trj::AbstractVector{<:AbstractMatrix{<:Integer}}, cmaps::AbstractVector{<:AbstractArray}; U=I(length(data)))
    Ncoeff = size(U, 2)
    img_shape = size(cmaps[1])
    img_idx = CartesianIndices(img_shape)

    dataU = similar(data[1], img_shape..., Ncoeff)
    xbp = zeros(eltype(data[1]), img_shape..., Ncoeff)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data[1], 2)
            dataU[img_idx, icoef] .= 0

            for it ∈ eachindex(data), is ∈ axes(data[it], 1), irep ∈ axes(data[it], 3)
                k_idx = ntuple(j -> mod1(trj[it][j, is] - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)
                dataU[k_idx, icoef] += data[it][is, icoil, irep] * conj(U[it, icoef, irep])
            end

            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end

function calculateBackProjection(data::AbstractVector{<:AbstractArray}, trj::AbstractVector{<:AbstractMatrix{<:Integer}}, img_shape::NTuple{N,Int}; U=I(length(data))) where {N}
    Ncoeff = size(U, 2)
    Ncoil = size(data[1], 2)
    img_idx = CartesianIndices(img_shape)

    dataU = similar(data[1], img_shape..., Ncoeff)
    xbp = zeros(eltype(data[1]), img_shape..., Ncoeff, Ncoil)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data[1], 2)
            dataU[img_idx, icoef] .= 0

            for it ∈ eachindex(data), is ∈ axes(data[it], 1), irep ∈ axes(data[it], 3)
                k_idx = ntuple(j -> mod1(trj[it][j, is] - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)
                dataU[k_idx, icoef] += data[it][is, icoil, irep] * conj(U[it, icoef, irep])
            end

            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef, icoil] .= fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end

function calculateCoilwiseCG(data::AbstractVector{<:AbstractArray{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
    Ncoil = size(data[1], 2)

    AᴴA = NFFTNormalOp(img_shape, trj, U[:, 1]; verbose)
    xbp = calculateBackProjection(data, trj, img_shape; U=U[:, 1], verbose)
    x = zeros(cT, img_shape..., Ncoil)

    for icoil ∈ axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter, verbose, reltol=0)
    end
    return x
end

function calculateCoilwiseCG(data::AbstractVector{<:CuArray{cT}}, trj::AbstractVector{<:CuArray{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
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

function calculateCoilwiseCG(data::AbstractVector{<:AbstractArray{cT}}, trj::AbstractVector{<:AbstractMatrix{<:Integer}}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=5, verbose=false) where {cT<:Complex,N}
    Ncoil = size(data[1], 2)

    AᴴA = FFTNormalOp(img_shape, trj, U[:, 1])
    xbp = calculateBackProjection(data, trj, img_shape; U=U[:, 1])
    x = zeros(cT, img_shape..., Ncoil)

    for icoil ∈ axes(xbp, length(img_shape) + 2)
        bi = vec(@view xbp[CartesianIndices(img_shape), 1, icoil])
        xi = vec(@view x[CartesianIndices(img_shape), icoil])
        cg!(xi, AᴴA, bi; maxiter, verbose) # maxiter <8 to avoid diverging
    end
    return x
end

## ##########################################################################
# Internal use
#############################################################################
function applyDensityCompensation!(data, trj; density_compensation=:radial_3D)
    if density_compensation == :radial_3D
        data .*= transpose(sum(abs2, trj, dims=1))
    elseif density_compensation == :radial_2D
        data .*= transpose(sqrt.(sum(abs2, trj, dims=1)))
    elseif density_compensation == :none
        # do nothing here
    elseif isa(density_compensation, AbstractVector{<:AbstractVector})
        data .*= reduce(hcat, density_compensation)
    else
        error("`density_compensation` can only be `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector}`")
    end
end

function test_dimension(data, trj, U, cmaps)
    Nt = size(U, 1)
    img_shape = size(cmaps)[1:end-1]
    Ncoil = size(cmaps)[end]

    Nt != size(data, 2) && ArgumentError(
        "The second dimension of data ($(size(data, 2))) and the first one of U ($Nt) do not match. Both should be number of time points.",
    )
    size(trj[1], 1) != length(img_shape) && ArgumentError(
        "`cmaps` contains $(length(img_shape)) image plus one coil dimension, yet the 1ˢᵗ dimension of each trj is of length $(size(trj,1)). They should match and reflect the dimensionality of the image (2D vs 3D).",
    )

    size(trj[1], 2) != size(data, 1) && ArgumentError(
        "The 2ⁿᵈ dimension of each `trj` is $(size(trj[1],2)) and the 1ˢᵗ dimension of `data` is $(size(data,1)). They should match and reflect the number of k-space samples.",
    )

    length(trj) != size(data, 2) && ArgumentError(
        "`trj` has the length $(length(trj)) and the 2ⁿᵈ dimension of data is $(size(data,2)). They should match and reflect the number of time points.",
    )

    Ncoil != size(data, 3) && ArgumentError(
        "The last dimension of `cmaps` is $(Ncoil) and the 3ⁿᵈ dimension of data is $(size(data,3)). They should match and reflect the number of coils.",
    )

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
