function calcCoilMaps(data, trj, U, img_shape::NTuple{N,Int}; kernel_size = ntuple(_->6, N), calib_size =  ntuple(_->24, N), eigThresh_1=0.04, eigThresh_2=0.0, nmaps=1) where {N}
    Ncoils = size(data,3)
    Ndims = length(img_shape)
    imdims = ntuple(i->i, Ndims)

    dataU = data .* U[:,1]'
    dataU .*= dropdims(sum(abs2, trj, dims=1), dims=1)
    dataU = reshape(dataU, :, size(dataU, 3))

    p = plan_nfft(reshape(trj, 3, :), img_shape; m=4, σ=2.0)
    xbp = Array{ComplexF32}(undef, img_shape..., Ncoils)

    img_idx = CartesianIndices(img_shape)

    @info "BP for coils maps: "
    @time begin
        @batch for ic ∈ 1:Ncoils
            @views mul!(xbp[img_idx,ic], adjoint(copy(p)), dataU[:,ic])
        end
    end

    kbp = fftshift(xbp, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    m = CartesianIndices(calib_size) .+ CartesianIndex((img_shape .- calib_size) .÷ 2)
    kbp = kbp[m,:]

    @info "espirit: "
    cmaps = @time MRIReco.espirit(kbp, img_shape, kernel_size, eigThresh_1=eigThresh_1, eigThresh_2=eigThresh_2, nmaps=nmaps)

    cmaps = [cmaps[img_idx,ic,1] for ic=1:Ncoils]
    xbp   = [  xbp[img_idx,ic  ] for ic=1:Ncoils]
    return cmaps, xbp
end