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
function calcCoilMaps(data::AbstractVector{<:AbstractMatrix{Complex{T}}}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U = ones(length(data)), density_compensation=:radial_3D, res_factor = 1, CG = false, mask_edges = false , kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    Ncoil = size(data[1], 2)
    Ndims = length(img_shape)
    imdims = ntuple(i -> i, Ndims)
    Nt = length(trj)

    if CG # reconstuct using CG
        DownSamplingFactor = minimum(img_shape .÷ (calib_size .* 2))
        img_shape_cmaps = img_shape.÷DownSamplingFactor

        trj_CG = Vector{Matrix{T}}(undef, Nt)
        data_CG = similar(data)
        for it ∈ eachindex(trj)
            trj_idx = [maximum(abs.(trj[it][:, i] .* DownSamplingFactor)) .< T(0.5) for i ∈ axes(trj[it], 2)]
            trj_CG[it] = trj[it][:, trj_idx] .* DownSamplingFactor
            data_CG[it] = data[it][trj_idx, :]
        end
        
        x = calculateCoilwiseCG(data_CG, trj_CG, img_shape_cmaps; U)

    else # reconstuct using filtered backprojection
        x = calculateBackProjection(data, trj, img_shape; U=U[:,1], density_compensation, verbose)
        x = dropdims(x,  dims=ndims(x)-1)
        img_shape_cmaps = img_shape
    end
    
    if mask_edges # rm edge artifacts in rosette CG recons
        x[1:2,:,:,:] .= 0
        x[end-1:end,:,:,:] .= 0
        x[:,1:2,:,:] .= 0
        x[:,end-1:end,:,:] .= 0
        x[:,:,1:2,:] .= 0
        x[:,:,end-1:end,:] .= 0
    end

    
    img_idx = CartesianIndices(round.(Int, img_shape ./ res_factor))


    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    m = CartesianIndices(calib_size) .+ CartesianIndex((img_shape_cmaps .- calib_size) .÷ 2)
    kbp = kbp[m, :]

    t = @elapsed begin
        cmaps = espirit(kbp, round.(Int, img_shape ./ res_factor), kernel_size, eigThresh_1=eigThresh_1, eigThresh_2=eigThresh_2, nmaps=nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[img_idx, ic, 1] for ic = 1:Ncoil]
    return cmaps
end

function calcCoilMaps(data::AbstractMatrix{Complex{T}}, trj::AbstractMatrix{T}, img_shape::NTuple{N,Int}; density_compensation=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    calcCoilMaps([data], [trj], img_shape; U = I(1), density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)
end