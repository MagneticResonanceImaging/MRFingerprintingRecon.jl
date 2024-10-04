"""
    calculateBackProjection(data, trj, img_shape; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps::AbstractVector{<:AbstractArray{cT,N}}; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps_img_shape; U, density_compensation, verbose)
    calculateBackProjection(data, trj, cmaps; U)

Calculate (filtered) backprojection

# Arguments
- `data <: Union{AbstractVector{<:AbstractMatrix{cT}},AbstractMatrix{cT}}`: Complex dataset either as AbstractVector of matrices or single matrix. The optional outer matrix defines different time frames that are reconstruced in the subspace defined in U.
- `trj <: Union{:AbstractVector{<:AbstractMatrix{T}},AbstractMatrix{T}}`: Trajectory with samples corresponding to the dataset either as AbstractVector of matrices or single matrix.
- `img_shape::NTuple{N,Int}`: Shape of image

# Optional Keyword Arguments
- `cmaps::::AbstractVector{<:AbstractArray{T}}`: Coil sensitivities as AbstractVector of arrays
- `cmaps_img_shape`: Either equal `img_shape` or `cmaps`
- `U::Matrix` = I(length(data)) or = I(1): Basis coefficients of subspace (only defined if data and trj have different timeframes)
- `density_compensation`=:`none`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector}`
- `verbose::Boolean`=`false`: Verbosity level

# Notes
- The type of the elements of the trajectory define if a gridded backprojection (eltype(trj[1]) or eltype(trj) <: Int) or a non-uniform (else) is performed.
"""
function calculateBackProjection(data::AbstractVector{<:AbstractArray{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    Ncoef = size(U, 2)

    trj_v = reduce(hcat, trj)
    p = plan_nfft(trj_v, img_shape; precompute=TENSOR, blocking=true, fftflags=FFTW.MEASURE)

    Ncoil = size(data[1], 2)
    xbp = Array{cT}(undef, img_shape..., Ncoef, Ncoil)

    trj_l = [size(trj[it], 2) for it in eachindex(trj)]
    data_temp = Vector{cT}(undef, sum(trj_l))

    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data[1], 2)
            @simd for it in eachindex(data)
                idx1 = sum(trj_l[1:it-1]) + 1
                idx2 = sum(trj_l[1:it])
                @views data_temp[idx1:idx2] .= data[it][:, icoil] .* conj(U[it, icoef])
            end
            applyDensityCompensation!(data_temp, trj_v; density_compensation)

            @views mul!(xbp[img_idx, icoef, icoil], adjoint(p), data_temp)
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractVector{<:AbstractMatrix{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, cmaps::AbstractVector{<:AbstractArray{cT,N}}; U=I(length(data)), density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T},N}
    test_dimension(data, trj, U, cmaps)

    trj_v = reduce(hcat, trj)
    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])

    p = plan_nfft(trj_v, img_shape; precompute=TENSOR, blocking=true, fftflags=FFTW.MEASURE)
    xbp = zeros(cT, img_shape..., Ncoef)
    xtmp = Array{cT}(undef, img_shape)

    trj_l = [size(trj[it], 2) for it in eachindex(trj)]
    data_temp = Vector{cT}(undef, sum(trj_l))
    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @simd for it in eachindex(data)
                idx1 = sum(trj_l[1:it-1]) + 1
                idx2 = sum(trj_l[1:it])
                @views data_temp[idx1:idx2] .= data[it][:, icoil] .* conj(U[it, icoef])
            end
            applyDensityCompensation!(data_temp, trj_v; density_compensation)
            mul!(xtmp, adjoint(p), data_temp)
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculateBackProjection(data::AbstractArray{cT}, trj::AbstractMatrix{T}, cmaps_img_shape; density_compensation=:none, verbose=false) where {T<:Real,cT<:Complex{T}}
    return calculateBackProjection([data], [trj], cmaps_img_shape; U=I(1), density_compensation, verbose)
end

# Method for GROG gridded data / trajectory
function calculateBackProjection(data::AbstractVector{<:AbstractArray}, trj::AbstractVector{<:AbstractMatrix{<:Integer}}, cmaps; U=I(length(data)))
    Ncoeff = size(U, 2)
    img_shape = size(cmaps[1])
    img_idx = CartesianIndices(img_shape)

    dataU = similar(data[1], img_shape..., Ncoeff)
    xbp = zeros(eltype(data[1]), img_shape..., Ncoeff)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data[1], 2)
            dataU[img_idx, icoef] .= 0

            for it ∈ eachindex(data), is ∈ axes(data[it], 1), irep ∈ axes(data[it], 3)
                k_idx = ntuple(j -> mod1(Int(trj[it][j, is]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)
                dataU[k_idx, icoef] += data[it][is, icoil, irep] * conj(U[it, icoef, irep])
            end

            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end


function calculateCoilwiseCG(data::AbstractVector{<:AbstractArray{cT}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U=I(length(data)), maxiter=100, verbose=false) where {T<:Real,cT<:Complex{T},N}
    Ncoil = size(data[1], 2)

    AᴴA = NFFTNormalOp(img_shape, trj, U[:, 1]; verbose)
    xbp = calculateBackProjection(data, trj, img_shape; U=U[:, 1], verbose)
    x = zeros(cT, img_shape..., Ncoil)

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
