"""
    calculate_backprojection(data, trj, img_shape; U, mask, density_compensation, verbose)
    calculate_backprojection(data, trj, cmaps::AbstractVector{<:AbstractArray{Tc,N}}; U, mask, density_compensation, verbose)
    calculate_backprojection(data, trj, cmaps_img_shape; U, mask, density_compensation, verbose)
    calculate_backprojection(data, trj, cmaps; U, mask)

Calculate (filtered) backprojection.

# Arguments
- `data <: AbstractArray{Complex}`: Complex dataset with axes (samples, time frames, channels). Time frames are reconstructed using the subspace defined in U. Use `CuArray` as input type to use CUDA GPU code.
- `trj <: AbstractArray`: Trajectory with sample coordinates corresponding to the dataset. For a Cartesian reconstruction, use `T <: Int` and define `trj[idim,it,ik] ∈ (1, img_shape[idim])`. If `T <: Float`, the NFFT is used. Use `CuArray` as input type to use CUDA code.

One of the following arguments needs to be supplied
- `img_shape::NTuple{N,Int}`: Shape of image; in this case, the data is reconstructed per coil.
- `cmaps::AbstractVector{<:AbstractArray{Tc}}`: Coil sensitivities; in this case, the coils are added up to a single backprojection. Use `AbstractVector{CuArray{Tc,N}}` as type for use with CUDA code.

# Optional Keyword Arguments
- `U::Matrix`=`I(size(trj)[end])` or `=I(1)`: Basis coefficients of subspace (only defined if data and trj have different timeframes)
- `mask::AbstractArray{Bool}`=`trues(size(trj)[2:end])`: Mask to indicate which k-space samples to use
- `density_compensation`=`:none`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type `AbstractVector{<:AbstractVector}`
- `verbose::Boolean`=`false`: Verbosity level
"""
function calculate_backprojection(data::AbstractArray{Tc,3}, trj::AbstractArray{T,3}, img_shape; U=I(size(trj)[end]), mask=trues(size(trj)[2:end]), density_compensation=:none, verbose=false) where {T <: Real, Tc <: Complex{T}}
    Ncoef = size(U, 2)

    # Count the number of samples per time frame using the mask
    nsamp_t = vec(sum(mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute backprojection."

    cumsum_nsamp = cumsum(nsamp_t) .+ 1
    prepend!(cumsum_nsamp, 1)

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
    trj_rs = trj[:, mask]
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs)) # transform matrix to tuples, change sign of FT exponent, change range to (0,2π)

    Ncoil = size(data, 3)
    xbp = Array{Tc}(undef, img_shape..., Ncoef, Ncoil)

    data_rs = data[mask, :]
    data_temp = Array{Tc}(undef, sum(mask))

    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ axes(data, 3)
            for it ∈ axes(data, 2)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @views data_temp[idx1:idx2] .= data_rs[idx1:idx2,icoil] .* conj(U[it,icoef])
            end
            apply_density_compensation!(data_temp, trj_rs; density_compensation)
            @views NonuniformFFTs.exec_type1!(xbp[img_idx, icoef, icoil], p, data_temp) # type 1: non-uniform points to uniform grid
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculate_backprojection(data::AbstractArray{Tc,3}, trj::AbstractArray{T,3}, cmaps::AbstractVector{<:AbstractArray{Tc,N}}; U=I(size(trj)[end]), mask=trues(size(trj)[2:end]), density_compensation=:none, verbose=false) where {T <: Real, Tc <: Complex{T}, N}
    test_dimension(data, trj, U, cmaps)

    Ncoef = size(U, 2)
    img_shape = size(cmaps[1])

    # Count the number of samples per time frame using the mask
    nsamp_t = vec(sum(mask; dims=1))
    @assert sum(nsamp_t) > 0 "Mask removes all samples, cannot compute backprojection."

    cumsum_nsamp = cumsum(nsamp_t) .+ 1 # start index of each frame
    prepend!(cumsum_nsamp, 1)

    p = PlanNUFFT(Complex{T}, img_shape; fftshift=true)
    trj_rs = trj[:, mask]
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj_rs))
    xbp = zeros(Tc, img_shape..., Ncoef)
    xtmp = Array{Tc}(undef, img_shape)

    data_rs = data[mask, :]
    data_temp = Array{Tc}(undef, sum(mask))

    img_idx = CartesianIndices(img_shape)
    verbose && println("calculating backprojection...")
    flush(stdout)
    for icoef ∈ axes(U, 2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
             @simd for it ∈ axes(data, 2)
                idx1 = cumsum_nsamp[it]
                idx2 = cumsum_nsamp[it + 1] - 1
                @views data_temp[idx1:idx2] .= data_rs[idx1:idx2,icoil] .* conj(U[it,icoef])
            end
            apply_density_compensation!(data_temp, trj_rs; density_compensation)
            NonuniformFFTs.exec_type1!(xtmp, p, data_temp) # type 1: non-uniform points to uniform grid
            xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s")
        flush(stdout)
    end
    return xbp
end

function calculate_backprojection(data::AbstractArray{Tc}, trj::AbstractArray{<:Integer,3}, cmaps::AbstractVector{<:AbstractArray}; U=I(size(trj)[end]), mask=trues(size(trj)[2:end])) where {Tc <: Complex}
    Ncoeff = size(U, 2)
    img_shape = size(cmaps[1])
    img_idx = CartesianIndices(img_shape)

    dataU = similar(data, img_shape..., Ncoeff)
    xbp = zeros(eltype(data), img_shape..., Ncoeff)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data, 3)
            dataU[img_idx, icoef] .= 0
            for it ∈ axes(data, 2), is ∈ axes(data, 1)
                if mask[is, it] # only incorporate samples within the mask
                    k_idx = ntuple(j -> mod1(trj[j, is, it] - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                    k_idx = CartesianIndex(k_idx)
                    for irep ∈ axes(data, 4)
                        dataU[k_idx, icoef] += data[is, it, icoil, irep] * conj(U[it, icoef, irep])
                    end
                end
            end
            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end

function calculate_backprojection(data::AbstractArray{Tc}, trj::AbstractArray{<:Integer,3}, img_shape; U=I(size(trj)[end]), mask=trues(size(trj)[2:end])) where {Tc <: Complex}
    Ncoeff = size(U, 2)
    Ncoil = size(data, 3)
    img_idx = CartesianIndices(img_shape)

    dataU = similar(data, img_shape..., Ncoeff)
    xbp = zeros(eltype(data), img_shape..., Ncoeff, Ncoil)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data, 3)
            dataU[img_idx, icoef] .= 0
            for it ∈ axes(data, 2), is ∈ axes(data, 1)
                if mask[is, it] # only incorporate samples within the mask
                    k_idx = ntuple(j -> mod1(trj[j, is, it] - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                    k_idx = CartesianIndex(k_idx)
                    for irep ∈ axes(data, 4)
                        dataU[k_idx, icoef] += data[is, it, icoil, irep] * conj(U[it, icoef, irep])
                    end
                end
            end
            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef, icoil] .= fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end

# wrappers for use with 4D arrays where the nr of ADC samples per readout is within a separate 2ⁿᵈ axis
function calculate_backprojection(data::AbstractArray{Tc,4}, trj::AbstractArray{T,4}, arg3; mask=trues(size(trj)[2:end]), kwargs...) where {T, Tc <: Complex}
    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return calculate_backprojection(data, trj, arg3; mask, kwargs...)
end

## ##########################################################################
# Internal helper functions
#############################################################################
function apply_density_compensation!(data, trj; density_compensation=:radial_3D)
    if density_compensation == :radial_3D
       data .*= vec(sum(abs2, trj, dims=1))
    elseif density_compensation == :radial_2D
        data .*= vec(sqrt.(sum(abs2, trj, dims=1)))
    elseif density_compensation == :none
        # do nothing here
    elseif isa(density_compensation, AbstractArray)
        data .*= density_compensation
    else
        error("`density_compensation` can only be `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractArray`")
    end
end

function test_dimension(data, trj, U, cmaps)
    Nt = size(U, 1)
    img_shape = size(cmaps[1])
    Ncoil = length(cmaps)

    Nt != size(data, 2) && throw(ArgumentError(
        "The second dimension of data ($(size(data, 2))) and the first one of U ($Nt) do not match. Both should be number of time points.",
    ))
    size(trj, 1) != length(img_shape) && throw(ArgumentError(
        "`cmaps` contains $(length(img_shape)) image plus one coil dimension, yet the 1ˢᵗ dimension of each trj is of length $(size(trj,1)). They should match and reflect the dimensionality of the image (2D vs 3D).",
    ))
    size(trj, 2) != size(data, 1) && throw(ArgumentError(
        "The 2ⁿᵈ dimension of each `trj` is $(size(trj,2)) and the 1ˢᵗ dimension of `data` is $(size(data,1)). They should match and reflect the number of k-space samples.",
    ))
    size(trj, 3) != size(data, 2) && throw(ArgumentError(
        "`trj` contains $(size(trj, 3)) time frames, while data consists of $(size(data,2)). They should match and reflect the number of time points.",
    ))
    Ncoil != size(data, 3) && throw(ArgumentError(
        "The last dimension of `cmaps` is $(Ncoil) and the 3ⁿᵈ dimension of data is $(size(data,3)). They should match and reflect the number of coils.",
    ))
end