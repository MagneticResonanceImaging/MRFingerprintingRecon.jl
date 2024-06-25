## ##########################################################################
# FFTNormalOp
#############################################################################

"""
    FFTNormalOp(img_shape, trj, U; cmaps)
    FFTNormalOp(M, U; cmaps)
    FFTNormalOp(Λ; cmaps)

Create normal operator of FFT operator.
Differentiate between functions exploiting a pre-calculated kernel basis `Λ` and the functions which calculate Λ based on a passed trajectory `trj` or mask `M`.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `U::Matrix{ComplexF32}`=(1,): Basis coefficients of subspace
- `cmaps::Matrix{ComplexF32}`: Coil sensitivities
- `M::Vector{Matrix{Float32}}`: Mask
- `Λ::Array{Complex{T},3}`: Toeplitz kernel basis
-  `num_fft_threads::Int` = `round(Int, Threads.nthreads()/size(U, 2))` or `round(Int, Threads.nthreads()/size(Λ, 1)): Number of Threads for FFT
"""
function FFTNormalOp(img_shape, trj, U; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)), verbose = false)
    Λ = calculateKernelBasis(img_shape, trj, U)
    return FFTNormalOp(Λ; cmaps, num_fft_threads)
end

function FFTNormalOp(M, U; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)), verbose = false)
    Λ = calculateKernelBasis(M, U)
    return FFTNormalOp(Λ; cmaps, num_fft_threads )
end

function FFTNormalOp(Λ; cmaps=(1,), num_fft_threads = round(Int, Threads.nthreads()/size(Λ, 1)), verbose = false)
    Ncoeff = size(Λ, 1)
    img_shape = size(Λ)[3:end]
    kL1 = similar(Λ, eltype(Λ), (img_shape..., Ncoeff))
    kL2 = similar(kL1)

    @views kmask = (Λ[1, 1, CartesianIndices(img_shape)] .!= 0)
    @allowscalar kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:, :, kmask_indcs]

     # GPU
     if typeof(Λ) <: AbstractGPUArray
        # GPU first to avoid Λ::JLArrays (<: AbstractArray & <: AbstractGPUArray) runs through CPU only
        # FFT interface supports differing options on CPU and GPU
        verbose && println("FFT: Abstract GPU Array version")
        ktmp = kL1[CartesianIndices(size(kL1))]
        fftplan! = plan_fft!(ktmp, [1, 2])
        ifftplan! = plan_ifft!(ktmp, [1, 2])

    # CPU
    else typeof(Λ) <: AbstractArray
        verbose && println("FFT: Abstract CPU Array version")
        ktmp = @view kL1[CartesianIndices(img_shape), 1]
        fftplan! = plan_fft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
        ifftplan! = plan_ifft!(ktmp; flags=FFTW.MEASURE, num_threads=num_fft_threads)
    end
    
    A = _FFTNormalOp(img_shape, Ncoeff, fftplan!, ifftplan!, Λ, kmask_indcs, kL1, kL2, cmaps)

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

## ##########################################################################
# Internal use
#############################################################################

struct _FFTNormalOp{S,E,F,G,H,I,J,K}
    shape::S
    Ncoeff::Int
    fftplan!::E
    ifftplan!::F
    Λ::G
    kmask_indcs::H
    kL1::I
    kL2::J
    cmaps::K
end

function calculateKernelBasis(img_shape, trj, U)
    Ncoeff = size(U, 2)
    Λ = zeros(eltype(U), Ncoeff, Ncoeff, img_shape...)
    for it ∈ axes(U, 1)
        for ix ∈ axes(trj[it], 2)
            k_idx = ntuple(j -> mod1(Int(trj[it][j, ix]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
            k_idx = CartesianIndex(k_idx)
            for ic ∈ CartesianIndices((Ncoeff, Ncoeff)), irep ∈ axes(U, 3)
                Λ[ic[1], ic[2], k_idx] += conj(U[it, ic[1], irep]) * U[it, ic[2], irep]
            end
        end
    end
    return Λ
end

function calculateKernelBasis(M, U)
    Ncoeff = size(U, 2)
    img_shape = size(M)[1:end-1]
    Λ = similar(U, eltype(U), (Ncoeff, Ncoeff, img_shape...))

    M .= ifftshift(M, 1:length(img_shape))
    Threads.@threads for i ∈ CartesianIndices(img_shape)
        Λ[:, :, i] .= U' * (M[i, :] .* U) #U' * diagm(D) * U
    end

    return Λ
end

# CPU
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
                @views S.fftplan! * S.kL1[idx, i]
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
                @views S.ifftplan! * S.kL2[idx, i]
                @views xr[idx, i] .+= α .* conj.(cmap) .* S.kL2[idx, i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end

# GPU
function LinearAlgebra.mul!(x::aT, S::_FFTNormalOp, b, α, β) where {aT <: AbstractGPUArray}
    idx = CartesianIndices(S.shape)

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    # Determine best threads and blocks
    max_threads = 256
    threads_x = min(max_threads, length(S.kmask_indcs))
    threads_y = min(max_threads ÷ threads_x, S.Ncoeff)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (length(S.kmask_indcs), S.Ncoeff) ./ threads)

    for cmap ∈ S.cmaps
        @views S.kL1[idx, :] .= cmap .* b[idx, :]
        S.fftplan! * S.kL1

        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff) .= 0
        @cuda threads=threads blocks=blocks kernel_mul!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs)

        S.ifftplan! * S.kL2
        @views xr[idx, :] .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end
    return x
end