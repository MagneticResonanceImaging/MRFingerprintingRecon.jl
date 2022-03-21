function calculateBackProjection(data::Array{T}, trj, U, cmaps) where {T}
    test_dimension(data, trj, U, cmaps)
    print("Calculating backprojection: ")
    @time begin
        Nt, Ncoef = size(U)
        img_shape = size(cmaps[1])
        Ncoils = length(cmaps)

        FFTW.set_num_threads(1)
        p = NFFT.NFFTPlan(trj[1], img_shape; fftflags = FFTW.MEASURE)
        pv = [copy(p) for _ = 1:Threads.nthreads()]
        xbp = [zeros(T, img_shape..., Ncoef) for _ = 1:Threads.nthreads()]
        xtmp = [Array{T}(undef, img_shape) for _ = 1:Threads.nthreads()]

        @batch for it ∈ 1:Nt
            tid = Threads.threadid()
            Ui = reshape(conj.(U[it, :]), one.(img_shape)..., Ncoef)
            NFFT.nodes!(pv[tid], trj[it])

            for icoil ∈ 1:Ncoils
                @views NFFT.nfft_adjoint!(pv[tid], data[:, it, icoil], xtmp[tid])
                @views xbp[tid] .+= conj.(cmaps[icoil]) .* xtmp[tid] .* Ui
            end
        end
    end
    return sum(xbp)
end

function test_dimension(data, trj, U, cmaps)
    Nt, _ = size(U)
    img_shape = size(cmaps[1])
    Ncoils = length(cmaps)

    Nt != size(data, 2) && ArgumentError("The second dimension of data ($(size(data, 2))) and the first one of U ($Nt) do not match. Both should be number of time points.")
    size(trj[1], 1) != length(img_shape) && ArgumentError("`cmaps` contains $(length(img_shape)) image plus one coil dimension, yet the 1ˢᵗ dimension of each trj is of length $(size(trj,1)). They should match and reflect the dimensionality of the image (2D vs 3D).")

    size(trj[1], 2) != size(data, 1) && ArgumentError("The 2ⁿᵈ dimension of each `trj` is $(size(trj[1],2)) and the 1ˢᵗ dimension of `data` is $(size(data,1)). They should match and reflect the number of k-space samples.")

    length(trj) != size(data, 2) && ArgumentError("`trj` has the length $(length(trj)) and the 2ⁿᵈ dimension of data is $(size(data,2)). They should match and reflect the number of time points.")

    Ncoils != size(data, 3) && ArgumentError("The last dimension of `cmaps` is $(Ncoils) and the 3ⁿᵈ dimension of data is $(size(data,3)). They should match and reflect the number of coils.")

end