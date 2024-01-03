function calculateKernelBasis(img_shape, trj, U)
    Ncoeff = size(U, 2)
    Nt = length(trj) # number of time points
    @assert Nt == size(U, 1) "Mismatch between trajectory and basis"

    Λ = zeros(eltype(U), Ncoeff, Ncoeff, img_shape...)

    for it ∈ eachindex(trj), ix ∈ axes(trj[it], 2)
        k_idx = ntuple(j -> mod1(Int(trj[it][j, ix]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
        k_idx = CartesianIndex(k_idx)

        for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
            Λ[ic[1], ic[2], k_idx] += conj(U[it, ic[1]]) * U[it, ic[2]]
        end
    end
    return Λ
end

function calculateKernelBasis(D, U)
    Ncoeff = size(U, 2)
    img_shape = size(D)[1:end-1]
    Λ = Array{eltype(U)}(undef, Ncoeff, Ncoeff, img_shape...)

    Threads.@threads for i ∈ CartesianIndices(img_shape)
        Λ[:, :, i] .= U' * (D[i, :] .* U) #U' * diagm(D) * U
    end
    Λ .= ifftshift(Λ, 3:(3+length(img_shape)-1)) #could fftshift D first

    return Λ
end

## ##########################################################################
# FFTNormalOpBasis
#############################################################################
struct _FFTNormalOpBasis{S,T,N,E,F,G}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::Array{Complex{T},3}
    kmask_indcs::Vector{Int}
    kL1::Array{Complex{T},N}
    kL2::Array{Complex{T},N}
    cmaps::G
end

function FFTNormalOpBasis(img_shape, U, trj; cmaps=(1,))
    Λ = calculateKernelBasis(img_shape, trj, U)
    return FFTNormalOpBasis(Λ; cmaps)
end

function FFTNormalOpBasis(D, U; cmaps=(1,))
    Λ = calculateKernelBasis(D, U)
    return FFTNormalOpBasis(Λ; cmaps)
end

function FFTNormalOpBasis(Λ; cmaps=(1,))
    Ncoeff = size(Λ, 1)
    img_shape = size(Λ)[3:end]
    kL1 = Array{eltype(Λ)}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    @views kmask = (Λ[1, 1, CartesianIndices(img_shape)] .!= 0)
    kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:, :, kmask_indcs]

    ktmp = @view kL1[CartesianIndices(img_shape), 1]
    fftplan = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=round(Int, Threads.nthreads() / Ncoeff))
    ifftplan = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=round(Int, Threads.nthreads() / Ncoeff))
    A = _FFTNormalOpBasis(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps)

    return LinearOperator(
        eltype(Λ),
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

function LinearAlgebra.mul!(x::Vector{T}, S::_FFTNormalOpBasis, b, α, β) where {T}
    idx = CartesianIndices(S.shape)

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        fill!(x, zero(T)) # to avoid 0 * NaN == NaN
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        for cmap ∈ S.cmaps
            Threads.@threads for i ∈ 1:S.Ncoeff # multiply by C and F
                @views S.kL1[idx, i] .= cmap .* b[idx, i]
                @views S.fftplan * S.kL1[idx, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i in eachindex(kL2_rs)
                kL2_rs[i] = 0
            end
            Threads.@threads for i ∈ axes(S.Λ, 3)
                @views @inbounds mul!(kL2_rs[S.kmask_indcs[i], :], S.Λ[:, :, i], kL1_rs[S.kmask_indcs[i], :])
            end

            Threads.@threads for i ∈ 1:S.Ncoeff # multiply by C' and F'
                @views S.ifftplan * S.kL2[idx, i]
                @views xr[idx, i] .+= α .* conj.(cmap) .* S.kL2[idx, i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end