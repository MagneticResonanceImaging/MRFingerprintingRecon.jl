function calculateBackProjection(data::AbstractArray{T}, trj, U, cmaps; verbose = false) where {T}
    test_dimension(data, trj, U, cmaps)

    _, Ncoef = size(U)
    img_shape = size(cmaps[1])

    p = plan_nfft(reduce(hcat,trj), img_shape; precompute=TENSOR, blocking = true, fftflags = FFTW.MEASURE)
    xbp = zeros(T, img_shape..., Ncoef)
    xtmp = Array{T}(undef, img_shape)

    dataU = similar(@view data[:,:,1]) # size = Ncycles*Nr x Nt
    img_idx = CartesianIndices(img_shape)
    for icoef ∈ axes(U,2)
        t = @elapsed for icoil ∈ eachindex(cmaps)
            @simd for i ∈ CartesianIndices(dataU)
                @inbounds dataU[i] = data[i,icoil] * conj(U[i[2],icoef])
            end
            mul!(xtmp, adjoint(p), vec(dataU))
            xbp[img_idx,icoef] .+= conj.(cmaps[icoil]) .* xtmp
        end
        verbose && println("coefficient = $icoef: t = $t s"); flush(stdout)
    end
    return xbp
end

function test_dimension(data, trj, U, cmaps)
    Nt, _ = size(U)
    img_shape = size(cmaps[1])
    Ncoils = length(cmaps)

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

    Ncoils != size(data, 3) && ArgumentError(
        "The last dimension of `cmaps` is $(Ncoils) and the 3ⁿᵈ dimension of data is $(size(data,3)). They should match and reflect the number of coils.",
    )

end
