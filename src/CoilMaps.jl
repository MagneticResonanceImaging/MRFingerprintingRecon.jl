"""
    calcCoilMaps(data, trj, img_shape; U, density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)

Estimate coil sensitivity maps using ESPIRiT [1].

# Arguments
- `data::AbstractVector{<:AbstractMatrix{Complex{T}}}`: Complex dataset either as AbstractVector of matrices or single matrix. The optional outer vector defines different time frames that are combined using the subspace defined in `U`
- `trj::AbstractVector{<:AbstractMatrix{T}}`: Trajectory with samples corresponding to the dataset either as AbstractVector of matrices or single matrix.
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
function calcCoilMaps(data::AbstractVector{<:AbstractMatrix{Complex{T}}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U = ones(length(data)), density_compensation=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    Ncoil = size(data[1], 2)
    Ndims = length(img_shape)
    imdims = ntuple(i -> i, Ndims)

    xbp = calculateBackProjection(data, trj, img_shape; U=U[:,1], density_compensation, verbose)
    xbp = dropdims(xbp, dims=ndims(xbp)-1)

    img_idx = CartesianIndices(img_shape)
    kbp = fftshift(xbp, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    m = CartesianIndices(calib_size) .+ CartesianIndex((img_shape .- calib_size) .÷ 2)
    kbp = kbp[m, :]

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size, eigThresh_1=eigThresh_1, eigThresh_2=eigThresh_2, nmaps=nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[img_idx, ic, 1] for ic = 1:Ncoil]
    return cmaps
end

function calcCoilMaps(data::AbstractMatrix{Complex{T}}, trj::AbstractMatrix{T}, img_shape::NTuple{N,Int}; density_compensation=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    calcCoilMaps([data], [trj], img_shape; U = I(1), density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)
end