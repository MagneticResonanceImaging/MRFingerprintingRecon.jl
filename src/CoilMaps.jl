function calcCoilMaps(data::AbstractArray{Complex{T},3}, trj::AbstractVector{<:AbstractMatrix{T}}, img_shape::NTuple{N,Int}; U = N==3 ? I(size(data,2)) : I(1), density_compensation=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    Ncoils = size(data, 3)
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

    cmaps = [cmaps[img_idx, ic, 1] for ic = 1:Ncoils]
    return cmaps
end
