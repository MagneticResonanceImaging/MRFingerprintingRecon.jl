## ##########################################################################
# FFTNormalOp
#############################################################################

"""
    FFTNormalOp(img_shape, trj, U; cmaps)
    FFTNormalOp(M, U; cmaps)
    FFTNormalOp(Λ; cmaps, eltype_x)

Create normal operator of FFT operator.
Differentiate between functions exploiting a pre-calculated kernel basis `Λ` and the functions which calculate Λ based on a passed trajectory `trj` or mask `M`.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `U::Matrix{ComplexF32}`=(1,): Basis coefficients of subspace
- `cmaps::Matrix{ComplexF32}`: Coil sensitivities
- `M::Vector{Matrix{Float32}}`: Mask
- `Λ::Array{Complex{T},3}`: Toeplitz kernel basis
- `num_fft_threads::Int = round(Int, Threads.nthreads()/size(U, 2))` or `round(Int, Threads.nthreads()/size(Λ, 1)): Number of Threads for FFT
- `eltype_x=eltype(Λ)` define the type of `x` (in the product `FFTNormalOp(Λ) * x`). The default is the same eltype as `Λ`
"""
function FFTNormalOp(img_shape, trj, U; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)))
    Λ = calculateKernelBasis(img_shape, trj, U)
    return FFTNormalOp(Λ; cmaps, num_fft_threads)
end

function FFTNormalOp(M, U; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)))
    Λ = calculateKernelBasis(M, U)
    return FFTNormalOp(Λ; cmaps, num_fft_threads )
end

function FFTNormalOp(Λ; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(Λ, 1)), eltype_x=eltype(Λ))
    Ncoeff = size(Λ, 1)
    img_shape = size(Λ)[3:end]
    kL1 = Array{eltype_x}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    @views kmask = (Λ[1, 1, CartesianIndices(img_shape)] .!= 0)
    kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:, :, kmask_indcs]

    ktmp = @view kL1[CartesianIndices(img_shape), 1]
    fftplan = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    A = _FFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps)

    return LinearOperator(
        eltype_x,
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

## ##########################################################################
# Internal use
#############################################################################
struct _FFTNormalOp{S,ΛType,T,N,E,F,G}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::ΛType
    kmask_indcs::Vector{Int}
    kL1::Array{Complex{T},N}
    kL2::Array{Complex{T},N}
    cmaps::G
end

function calculateKernelBasis(img_shape, trj, U)
    Ncoeff = size(U, 2)
    Λ = zeros(eltype(U), Ncoeff, Ncoeff, img_shape...)

    Threads.@threads for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
        for it ∈ axes(U, 1), ix ∈ axes(trj[it], 2), irep ∈ axes(U, 3)
            k_idx = ntuple(j -> mod1(Int(trj[it][j, ix]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
            k_idx = CartesianIndex(k_idx)
            Λ[ic[1], ic[2], k_idx] += conj(U[it, ic[1], irep]) * U[it, ic[2], irep]
        end
    end
    return Λ
end

function calculateKernelBasis(M, U)
    Ncoeff = size(U, 2)
    img_shape = size(M)[1:end-1]
    Λ = Array{eltype(U)}(undef, Ncoeff, Ncoeff, img_shape...)

    M .= ifftshift(M, 1:length(img_shape))
    Threads.@threads for i ∈ CartesianIndices(img_shape)
        Λ[:, :, i] .= U' * (M[i, :] .* U) #U' * diagm(D) * U
    end

    return Λ
end

function LinearAlgebra.mul!(x::AbstractVector{T}, S::_FFTNormalOp, b, α, β) where {T}
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
