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

function calculateBackProjection(data::AbstractArray{T}, trj, U, cmaps::AbstractVector{<:AbstractArray{T}}; density_compensation=:none, verbose=false) where T
    @warn "calculateBackProjection(data, trj, U, cmaps) has been deprecated – call calculateBackProjection(data, trj, cmaps; U=U) with U as a keyword argument instead." maxlog=1
    return calculateBackProjection(data, trj, cmaps; U, density_compensation, verbose)
end


function calculateBackProjection(data::AbstractArray{T,N}, trj, cmaps::AbstractVector{<:AbstractArray{T}}; U = N==3 ? I(size(data,2)) : I(1), density_compensation=:none, verbose=false) where {N,T}
    if typeof(trj) <: AbstractMatrix
        trj = [trj]
    end

    if ndims(data) == 2
        data = reshape(data, size(data, 1), 1, size(data, 2))
    end

    test_dimension(data, trj, U, cmaps)

    Ncoef = size(U,2)
    img_shape = size(cmaps[1])

    p = plan_nfft(reduce(hcat,trj), img_shape; precompute=TENSOR, blocking = true, fftflags = FFTW.MEASURE)
    xbp = zeros(T, img_shape..., Ncoef)
    xtmp = Array{T}(undef, img_shape)

    data_temp = similar(@view data[:,:,1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection..."); flush(stdout)
    for icoef ∈ axes(U,2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @simd for i ∈ CartesianIndices(data_temp)
                @inbounds data_temp[i] = data[i,icoil] * conj(U[i[2],icoef])
            end
            applyDensityCompensation!(data_temp, trj; density_compensation)

            mul!(xtmp, adjoint(p), vec(data_temp))
            xbp[img_idx,icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

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
