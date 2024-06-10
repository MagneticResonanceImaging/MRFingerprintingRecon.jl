## ##########################################################################
# NFFTNormalOp
#############################################################################

"""
    NFFTNormalOp(img_shape, trj, U; cmaps, verbose, num_fft_threads)

Create normal operator of NFFT operator.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `U::Matrix{ComplexF32}`: Basis coefficients of subspace
- `cmaps::Matrix{ComplexF32}`: Coil sensitivities
- `verbose::Boolean`: Verbose level
- `num_fft_threads::Int`: Number of threads for FFT
"""
function NFFTNormalOp(img_shape, trj, U;
    cmaps=[ones(size(trj[1],2), img_shape)],
    verbose = false,
    num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)),
    )

    Λ, kmask_indcs = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; verbose=verbose)

    @assert length(kmask_indcs) == size(Λ,3) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))

    Ncoeff = size(Λ, 1)
    img_shape_os = 2 .* img_shape
    kL1 = similar(Λ, eltype(Λ), (img_shape_os..., Ncoeff))
    kL2 = similar(kL1)

    # GPU
    if typeof(Λ) <: AbstractGPUArray
        verbose && println("GPU Operator")
        # ktmp = kL1[CartesianIndices(img_shape_os)]
        ktmp = kL1[CartesianIndices(size(kL1))]
        fftplan!  = plan_fft!(ktmp, [1, 2])
        ifftplan! = plan_ifft!(ktmp, [1, 2])

     # CPU
    else typeof(Λ) <: AbstractArray
        verbose && println("CPU Operator")
        ktmp = @view kL1[CartesianIndices(img_shape_os),1]
        fftplan!  = plan_fft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)
        ifftplan! = plan_ifft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)
    end

    A = _NFFTNormalOp(img_shape, Ncoeff, fftplan!, ifftplan!, Λ, kmask_indcs, kL1, kL2, cmaps)

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

struct _NFFTNormalOp{S,E,F,G,H,I,J,K}
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

function calculate_kmask_indcs(img_shape_os, trj)

    # GPU
    if typeof(trj[1]) <: AbstractGPUArray
        nfftplan! = plan_nfft(CuArray, reduce(hcat, trj), img_shape_os; m=5, σ=1)
        convolve_transpose!(nfftplan!, CuArray(ones(Complex{eltype(trj[1])}, size(nfftplan!)[1])), nfftplan!.tmpVec)

    # CPU
    else
        nfftplan! = plan_nfft(Array, reduce(hcat, trj), img_shape_os; precompute = POLYNOMIAL, blocking = false, fftflags = FFTW.MEASURE, m=5, σ=1)
        convolve_transpose!(nfftplan!, ones(Complex{eltype(trj[1])}, size(nfftplan!)[1]), nfftplan!.tmpVec)
    end
    kmask = (nfftplan!.tmpVec .!= 0)
    @allowscalar kmask_indcs = findall(vec(kmask))
    return kmask_indcs
end

function kernel_mul2!(S, Uc, U, ic1, ic2)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if i <= size(U, 1) && j <= size(S, 1)
            S[j,i] = Uc[i,ic1] * U[i,ic2]
    end
    return
end

function kernel_sort!(Λ, λ, λ3, kmask_indcs, ic1, ic2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    if i <= length(kmask_indcs)
        Λ[ic2,ic1,i] = λ3[kmask_indcs[i]]
        Λ[ic1,ic2,i] =  λ[kmask_indcs[i]]
    end
    return
end

function calculateToeplitzKernelBasis(img_shape_os, trj, U; verbose = false)

    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)
    Nt = size(U,1)
    Nk = size(trj[1],2)

    λ  = similar(U, eltype(U), (img_shape_os))
    λ2 = similar(λ)
    λ3 = similar(λ)
    Λ = similar(U, eltype(U), (Ncoeff, Ncoeff, length(kmask_indcs)))
    S = similar(U, eltype(U), (Nk, Nt))

    # GPU
    if typeof(U) <: AbstractGPUArray
        verbose && println("GPU based Toeplitz Kernel Calculation")

        fftplan!  = plan_fft(λ)
        nfftplan! = plan_nfft(CuArray, reduce(hcat, trj), img_shape_os; m=5, σ=2)

        max_threads = 256
        # multiplication
        threads_x = min(max_threads, size(U, 1))
        threads_y = min(max_threads ÷ threads_x, size(S, 1))
        threads = (threads_x, threads_y)
        blocks = ceil.(Int, (size(U, 1), size(S, 1)) ./ threads)
        # sorting
        threads_sort = min(max_threads, length(kmask_indcs))
        blocks_sort = ceil.(Int, length(kmask_indcs) ./ threads_sort)

    # CPU
    else
        verbose && println("CPU based Toeplitz Kernel Calculation")

        fftplan!  = plan_fft(λ; flags = FFTW.MEASURE, num_threads=Threads.nthreads())
        nfftplan! = plan_nfft(Array, reduce(hcat, trj), img_shape_os; precompute = TENSOR, blocking = true, fftflags = FFTW.MEASURE, m=5, σ=2)
    end

    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin

                # GPU
                if typeof(U) <: AbstractGPUArray
                    @cuda threads=threads blocks=blocks kernel_mul2!(S, conj(U), U, ic1, ic2)

                # CPU
                else
                    @simd for it ∈ axes(U,1)
                        @inbounds S[:,it] .= conj(U[it,ic1]) * U[it,ic2]
                    end
                end

                mul!(λ, adjoint(nfftplan!), reshape(S, Nk * Nt))
                fftshift!(λ2, λ)
                mul!(λ, fftplan!, λ2)
                λ2 .= conj.(λ2)
                mul!(λ3, fftplan!, λ2)

                # GPU
                if typeof(U) <: AbstractGPUArray
                    @cuda threads=threads_sort blocks=blocks_sort kernel_sort!(Λ, λ, λ3, kmask_indcs, ic1, ic2)

                # CPU
                else
                    Threads.@threads for it ∈ eachindex(kmask_indcs)
                        @inbounds Λ[ic2,ic1,it] = λ3[kmask_indcs[it]]
                        @inbounds Λ[ic1,ic2,it] =  λ[kmask_indcs[it]]
                    end
                end
            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end

    return Λ, kmask_indcs
end

# CPU
function LinearAlgebra.mul!(x::AbstractVector{T}, S::_NFFTNormalOp, b, α, β) where {T}
    # Keep specialized on x::Vector, because function should be not called for x::JLArray,
    # which is member of AbstractArray and(!) AbstractGPUArray

    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)

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
            Threads.@threads for i ∈ 1:S.Ncoeff
                S.kL1[idxos, i] .= zero(T)
                @views S.kL1[idx, i] .= cmap .* b[idx, i]
                @views S.fftplan! * S.kL1[idxos, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i in eachindex(kL2_rs)
                kL2_rs[i] = 0
            end
            Threads.@threads for i ∈ axes(S.Λ, 3)
                @views @inbounds mul!(kL2_rs[S.kmask_indcs[i], :], S.Λ[:, :, i], kL1_rs[S.kmask_indcs[i], :])
            end

            Threads.@threads for i ∈ 1:S.Ncoeff
                @views S.ifftplan! * S.kL2[idxos, i]
                @views xr[idx,i] .+= α .* conj.(cmap) .* S.kL2[idx,i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end


# GPU
function kernel_mul_radial!(kL2_rs, Λ, kL1_rs, kmask_indcs)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if i <= length(kmask_indcs) && j <= size(kL2_rs, 2)
    
        ind = kmask_indcs[i]
        tmp = 0
    
        for k in 1:size(Λ, 2)
            tmp += Λ[j, k, i] * kL1_rs[ind, k]
        end
        kL2_rs[ind, j] = tmp
    end
    return
end

function LinearAlgebra.mul!(x::aT, S::_NFFTNormalOp, b, α, β) where {aT <: AbstractGPUArray}
    # Keep specialized on x::Vector, because function should be not called for x::JLArray,
    # which is member of AbstractArray and(!) AbstractGPUArray

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)

    # Determine best threads and blocks
    max_threads = 256
    threads_x = min(max_threads, length(S.kmask_indcs))
    threads_y = min(max_threads ÷ threads_x, S.Ncoeff)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (length(S.kmask_indcs), S.Ncoeff) ./ threads)

    for cmap ∈ S.cmaps
        S.kL1[idxos, :] .= 0
        @views S.kL1[idx, :] .= cmap .* b[idx, :]
        S.fftplan! * S.kL1

        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff) .= 0
        @cuda threads=threads blocks=blocks kernel_mul_radial!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs) # FIXME: Reuse cartesian function here

        S.ifftplan! * S.kL2
        @views xr[idx, :] .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end

    return x
end