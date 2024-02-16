function calculateBackProjection(data::AbstractArray{T}, trj, U, cmaps::AbstractVector{<:AbstractArray{T}}; density_compensation=:none, verbose=false) where T
    @warn "calculateBackProjection(data, trj, U, cmaps) has been deprecated – call calculateBackProjection(data, trj, cmaps; U=U) with U as a keyword argument instead." maxlog=1
    return calculateBackProjection(data, trj, cmaps; U, density_compensation, verbose)
end


function calcCoilMaps(data::AbstractArray{Complex{T},3}, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractMatrix{Complex{T}}, img_shape::NTuple{N,Int}; density_compensation=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    @warn "calcCoilMaps(data, trj, U, img_shape) has been deprecated – call calcCoilMaps(data, trj, img_shape; U=U) with U as a keyword argument instead." maxlog=1
    return calcCoilMaps(data, trj, img_shape; U, density_compensation, kernel_size, calib_size, eigThresh_1, eigThresh_2, nmaps, verbose)
end
