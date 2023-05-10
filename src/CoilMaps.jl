function calcCoilMaps(data::AbstractArray{Complex{T},3}, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractMatrix{Complex{T}}, img_shape::NTuple{N,Int}; density_compensation::Union{Symbol, <:AbstractVector{<:AbstractVector{T}}}=:radial_3D, kernel_size = ntuple(_->6, N), calib_size =  ntuple(_->24, N), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose = false) where {N,T}
    Ncoils = size(data,3)
    Ndims = length(img_shape)
    imdims = ntuple(i->i, Ndims)

    p = plan_nfft(reduce(hcat,trj), img_shape; precompute=TENSOR, blocking = true, fftflags = FFTW.MEASURE)
    xbp = Array{Complex{T}}(undef, img_shape..., Ncoils)

    dataU = similar(@view data[:,:,1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    t = @elapsed for icoil ∈ axes(data,3)
        if density_compensation == :radial_3D
            @simd for i ∈ CartesianIndices(dataU)
                dataU[i] = data[i,icoil] * conj(U[i[2],1]) * sum(abs2, @view trj[i[2]][:,i[1]])
            end
        elseif density_compensation == :radial_2D
            @simd for i ∈ CartesianIndices(dataU)
                dataU[i] = data[i,icoil] * conj(U[i[2],1]) * sqrt(sum(abs2, @view trj[i[2]][:,i[1]]))
            end
        elseif density_compensation == :none
            # no density compensation; premultiply data with inverse of sampling density before calling function
            @simd for i ∈ CartesianIndices(dataU)
                dataU[i] = data[i,icoil] * conj(U[i[2],1])
            end
        elseif isa(density_compensation, Symbol)
            error("`density_compensation` can only be `:radial_3D`, `:radial_2D`, `:none`, or of type  `AbstractVector{<:AbstractVector{T}}`")
        else
            @simd for i ∈ CartesianIndices(dataU)
                dataU[i] = data[i,icoil] * conj(U[i[2],1]) * density_compensation[i[2]][i[1]]
            end
        end

        @views mul!(xbp[img_idx,icoil], adjoint(p), vec(dataU))
    end
    verbose && println("BP for coils maps: $t s")

    kbp = fftshift(xbp, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    m = CartesianIndices(calib_size) .+ CartesianIndex((img_shape .- calib_size) .÷ 2)
    kbp = kbp[m,:]

    t = @elapsed begin
        cmaps = espirit(kbp, img_shape, kernel_size, eigThresh_1=eigThresh_1, eigThresh_2=eigThresh_2, nmaps=nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cmaps[img_idx,ic,1] for ic=1:Ncoils]
    xbp   = [  xbp[img_idx,ic  ] for ic=1:Ncoils]
    return cmaps, xbp
end