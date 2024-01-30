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


function calculateBackProjection(data::AbstractArray{T}, trj, img_shape; U=ones(T, 1, 1), density_compensation=:radial_3D, verbose=false) where T
    if typeof(trj) <: AbstractMatrix
        trj = [trj]
    end

    if ndims(data) == 2
        data = reshape(data, size(data, 1), 1, size(data, 2))
    end

    Ncoils = size(data, 3)

    p = plan_nfft(reduce(hcat, trj), img_shape; precompute=TENSOR, blocking=true, fftflags=FFTW.MEASURE)
    xbp = Array{T}(undef, img_shape..., Ncoils)

    data_temp = similar(@view data[:, :, 1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    t = @elapsed for icoil ∈ axes(data, 3)
        data_temp .= data[:, :, icoil] .* U[:, 1]'

        applyDensityCompensation!(data_temp, trj; density_compensation)

        @views mul!(xbp[img_idx, icoil], adjoint(p), vec(data_temp))
    end
    verbose && println("Backprojection: $t s")
    return xbp
end


function calcCoilMaps(data::AbstractArray{Complex{T},3}, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractMatrix{Complex{T}}, img_shape::NTuple{N,Int}; density_compensation::Union{Symbol,<:AbstractVector{<:AbstractVector{T}}}=:radial_3D, kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(_ -> 24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    Ncoils = size(data, 3)
    Ndims = length(img_shape)
    imdims = ntuple(i -> i, Ndims)

    xbp = calculateBackProjection(data, trj, img_shape; U, density_compensation, verbose)

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
