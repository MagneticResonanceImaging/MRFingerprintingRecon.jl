"""
    calcCoilMaps(data, trj, img_shape; U, density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)

Estimate coil sensitivity maps using ESPIRiT [1].

# Arguments
- `data::AbstractArray{Complex{T}}`: Complex dataset with axes (samples, time frames, channels). Time frames are reconstructed using the subspace defined in U. . Use `CuArray` as input type to use CUDA code.
- `trj::AbstractArray{T}`: Trajectory with sample coordinates corresponding to the dataset Use `CuArray` as input type to use GPU code.
- `img_shape::NTuple{N,Int}`: Shape of image

# Keyword Arguments
- `U::Matrix` = N==3 ? ones(size(data,1)) : I(1): Basis coefficients of subspace (only defined if `data` and `trj` are vectors of matrices)
- `density_compensation`=:`radial_3D`: Values of `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector}`
- `kernel_size`=`ntuple(_ -> 6, N)`: Kernel size
- `calib_size`=`ntuple(_ -> 24, N)`: Size of calibration region
- `eigThresh_1`=0.01: Threshold of first eigenvalue
- `eigThresh_2`=0.9: Threshold of second eigenvalue
- `nmaps`=1: Number of estimated maps
- `verbose::Boolean`=`false`: Verbosity level

# return
- `cmaps::::Vector{<:Array{Complex{T}}}`: Coil sensitivities as Vector of arrays

# References
[1] Uecker, M., Lai, P., Murphy, M.J., Virtue, P., Elad, M., Pauly, J.M., Vasanawala, S.S. and Lustig, M. (2014), ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. Magn. Reson. Med., 71: 990-1001. https://doi.org/10.1002/mrm.24751
"""
function calculateCoilMaps(data::AbstractArray{Complex{T}}, trj::AbstractArray{T}, img_shape::NTuple{N,Int}; U=ones(Complex{T}, size(trj)[end]), mask=trues(size(trj)[2:end]), kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, Niter=100, verbose=false) where {N,T}
    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."
    
    calib_scale = img_shape ./ calib_size
    mask_calib = reshape(all(abs.(trj) .* calib_scale .< 0.5; dims=1), size(trj, 2), :)
    mask_calib .&= mask # new mask only takes data within calib region
    trj_calib = trj .* convert.(T, calib_scale) # scale trj for correct image dims

    x = calculateCoilwiseCG(data, trj_calib, calib_size; U, mask=mask_calib, Niter)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[CartesianIndices(img_shape), ic, in] for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

function calculateCoilMaps(data::AbstractArray{Complex{T}}, trj::AbstractArray{<:Integer}, img_shape::NTuple{N,Int}; U=ones(Complex{T}, size(data, 2)), mask=trues(size(trj)[2:end]), kernel_size=ntuple(_ -> 6, N), calib_size=img_shape .÷ (img_shape[1] ÷ 32), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, Niter=5, verbose=false) where {N,T}
    lower_bound = @. Int(ceil((img_shape - calib_size) / 2))
    upper_bound = @. lower_bound + calib_size + 1
    mask_calib = dropdims(all(trj .> lower_bound; dims=1) .& all(trj .< upper_bound; dims=1); dims=1)
    mask_calib .&= mask

    x = calculateCoilwiseCG(data, trj, calib_size; U, mask=mask_calib, Niter=Niter)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[CartesianIndices(img_shape), ic, in] for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

# wrappers for use with 4D arrays where the nr of ADC samples per readout is within a separate 2ⁿᵈ axis
function calculateCoilMaps(data::AbstractArray{cT,4}, trj::AbstractArray{T,4}, img_shape::NTuple{N,Int}; mask=trues(size(trj)[2:end]), kwargs...) where {T,cT<:Complex,N}
    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj, 1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return calculateCoilMaps(data, trj, img_shape; kwargs..., mask)
end