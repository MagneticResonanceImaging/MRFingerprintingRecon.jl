"""
    calculateBackProjection(data, trj, img_shape; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps; U, density_compensation, verbose)

Calculate backprojection

# Arguments
- `data::AbstractArray{T}`: Basis coefficients of subspace
- `trj::Vector{Matrix{Float32}}`: Trajectory
- `img_shape::NTuple{N,Int}`: Shape of image
- `U::Matrix{ComplexF32}`: Basis coefficients of subspace
- `density_compensation`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector}``
- `verbose::Boolean`: Verbosity level
- `cmaps::::AbstractVector{<:AbstractArray{T}}`: Coil sensitivities

"""
function calculateBackProjection(data::AbstractArray{T}, trj, img_shape::NTuple{N,Int}; U = N==3 ? I(size(data,2)) : I(1), density_compensation=:none, verbose=false) where {N,T}
    if typeof(trj) <: AbstractMatrix
        trj = [trj]
    end

    if ndims(data) == 2
        data = reshape(data, size(data, 1), 1, size(data, 2))
    end
    Ncoils = size(data, 3)
    Ncoef = size(U,2)

    p = plan_nfft(reduce(hcat, trj), img_shape; precompute=TENSOR, blocking=true, fftflags=FFTW.MEASURE)
    xbp = Array{T}(undef, img_shape..., Ncoef, Ncoils)

    data_temp = similar(@view data[:, :, 1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U,2)
        t = @elapsed for icoil ∈ axes(data, 3)
            @simd for i ∈ CartesianIndices(data_temp)
                @inbounds data_temp[i] = data[i,icoil] * conj(U[i[2],icoef])
            end
            applyDensityCompensation!(data_temp, trj; density_compensation)
            @views mul!(xbp[img_idx, icoef, icoil], adjoint(p), vec(data_temp))
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::aT, trj, cmaps::AbstractVector{aT2}; U = I(size(data,2)), density_compensation=:none, verbose=false) where {aT,aT2}

    if typeof(trj) <: Union{AbstractMatrix, AbstractGPUMatrix}
        trj = [trj]
    end

    if ndims(data) == 2
        data = reshape(data, size(data, 1), 1, size(data, 2))
    end

    test_dimension(data, trj, U, cmaps)

    Ncoef = size(U,2)
    img_shape = size(cmaps[1])

    # GPU
    if aT <: AbstractGPUArray
        verbose && println("GPU Backprojection")
        p = plan_nfft(CuArray, reduce(hcat,trj), img_shape) # CuNFFT requires k::Matrix{T}!
        xbp = CuArray(zeros(eltype(data), img_shape..., Ncoef))
        xtmp = CuArray{eltype(data)}(undef, img_shape)

        max_threads = 256
        threads_x = min(max_threads, size(data, 1))
        threads_y = min(max_threads ÷ threads_x, size(data, 2))
        threads = (threads_x, threads_y)
        blocks = ceil.(Int, (size(data, 1), size(data, 2)) ./ threads)

    # CPU
    else
        verbose && println("CPU Backprojection")
        p = plan_nfft(Array, reduce(hcat,trj), img_shape; precompute=TENSOR, blocking = true, fftflags = FFTW.MEASURE)
        xbp = zeros(eltype(data), img_shape..., Ncoef)
        xtmp = Array{eltype(data)}(undef, img_shape)
    end

    data_temp = similar(@view data[:,:,1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U,2)
        t = @elapsed for icoil ∈ eachindex(cmaps)

            # GPU
            if aT <: AbstractGPUArray
                @cuda threads=threads blocks=blocks kernel_mul!(data_temp, data, conj(U), icoef, icoil)

            # CPU
            else
                @simd for i ∈ CartesianIndices(data_temp)
                    @inbounds data_temp[i] = data[i,icoil] * conj(U[i[2],icoef])
                end
            end

            applyDensityCompensation!(data_temp, trj; density_compensation)

            mul!(xtmp, adjoint(p), reshape(data_temp, size(data,1)*size(data,2)))
            xbp[img_idx,icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

"""
    calculateBackProjection_gridded(data, trj, U, cmaps)

Calculate gridded backprojection

# Arguments
- `data::Matrix{ComplexF32}`: Basis coefficients of subspace
- `trj::Vector{Matrix{Float32}}`: Trajectory
- `U::Matrix{ComplexF32}`: Basis coefficients of subspace
- `cmaps::Matrix{ComplexF32}`: Coil sensitivities

# Note
In case of repeated sampling (Nrep > 1), a joint basis reconstruction is required.
Therefore, the basis needs to have a temporal dimension of Nt⋅Nrep with Nt as time dimension defined by the trajectory.
"""
function calculateBackProjection_gridded(data, trj, U, cmaps)
    Ncoil = length(cmaps)
    Ncoeff = size(U, 2)
    img_shape = size(cmaps[1])
    img_idx = CartesianIndices(img_shape)

    Nt = length(trj)
    Nrep = size(data, 4)

    if (1 != Nrep) # Avoid error during reshape that joins rep and t dim
        data = permutedims(data, (1,2,4,3))
        @assert Nt*Nrep == size(U, 1) "Mismatch between data and basis"
    else
        @assert Nt == size(U, 1) "Mismatch between trajectory and basis"
    end
    data = reshape(data, :, Nt*Nrep, Ncoil)

    dataU = similar(data, img_shape..., Ncoeff)
    xbp = zeros(eltype(data), img_shape..., Ncoeff)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data, 3)
            dataU[img_idx, icoef] .= 0

            for i ∈ CartesianIndices(@view data[:, :, 1, 1])
                t_idx = mod(i[2] + Nt - 1, Nt) + 1 # "mod" to incorporate repeated sampling pattern, "mod(i[2]+Nt-1,Nt)+1" to compensate for one indexing
                k_idx = ntuple(j -> mod1(Int(trj[t_idx][j, i[1]]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)

                @views dataU[k_idx, icoef] += data[i[1], i[2], icoil] * conj(U[i[2], icoef])
            end

            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end

## ##########################################################################
# Internal use
#############################################################################

function applyDensityCompensation!(data, trj; density_compensation=:radial_3D)
    for it in axes(data, 2)
        if density_compensation == :radial_3D
            data[:, it] .*= transpose(sum(abs2, trj[it], dims=1))
        elseif density_compensation == :radial_2D
            data[:, it] .*= transpose(sqrt.(sum(abs2, trj[it], dims=1)))
        elseif density_compensation == :none
            # do nothing here
        elseif isa(density_compensation, AbstractVector{<:AbstractVector})
            data[:, it] .*= density_compensation[it]
        else
            error("`density_compensation` can only be `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector}`")
        end
    end
end

function test_dimension(data, trj, U, cmaps)
    Nt = size(U,1)
    img_shape = size(cmaps)[1:end-1]
    Ncoils = size(cmaps)[end]

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

    Ncoils != size(data, 3) && ArgumentError(
        "The last dimension of `cmaps` is $(Ncoils) and the 3ⁿᵈ dimension of data is $(size(data,3)). They should match and reflect the number of coils.",
    )

end

function kernel_mul!(data_temp, data, Uc, icoef, icoil)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if i <= size(data, 1) && j <= size(data, 2)
        data_temp[i,j] = data[i,j,icoil] * Uc[j,icoef]
    end
    return
end
