function MRFingerprintingRecon.NFFTNormalOp(
    img_shape,
    trj::CuArray{T,3},
    U::CuArray{T};
    cmaps = [CUDA.ones(T, img_shape)],
    mask = CUDA.ones(Bool, size(trj)[2:end]),
    verbose = false
    ) where {T <: Real}
    Λ, kmask_indcs = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; mask, verbose)
    return NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps)
end

function NFFTNormalOp(
    img_shape,
    Λ::CuArray{T},
    kmask_indcs;
    cmaps=[CuArray(ones(T, img_shape))]
    ) where {T <: Real}
    @assert length(kmask_indcs) == size(Λ, length(size(Λ))) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))

    # derive Ncoeff from length of packed axis using quadratic eqn
    Ncoeff = (isqrt(8 * size(Λ, 1) + 1) - 1) ÷ 2

    img_shape_os = 2 .* img_shape
    kL1 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)

    fftplan  = plan_fft!( kL1, 1:length(img_shape_os))
    ifftplan = plan_ifft!(kL2, 1:length(img_shape_os))

    # indexing into the upper triangular matrix
    ind_lookup = CuArray([j<k ? j+k*(k-1)÷2 : k+j*(j-1)÷2 for j ∈ 1:Ncoeff, k ∈ 1:Ncoeff])

    # set up the threading for the GPU
    kL1_rs = reshape(kL1, :, Ncoeff)
    kL2_rs = reshape(kL2, :, Ncoeff)
    kernel = @cuda launch=false kernel_mul!(kL2_rs, Λ, kL1_rs, kmask_indcs, ind_lookup)
    config = launch_configuration(kernel.fun)

    threads_x = min(config.threads ÷ Ncoeff, length(kL2_rs))
    threads_y = min(config.threads, Ncoeff)
    threads = (threads_x, threads_y)
    blocks = cld.((length(kmask_indcs), Ncoeff), threads)

    # Set up the actual object
    A = MRFingerprintingRecon._NFFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps, ind_lookup, threads, blocks)

    return LinearOperator(
        Complex{T},
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β);
        S = CuArray{Complex{T}}
    )

end

#############################################################################
# Internal use
#############################################################################
function calculate_kmask_indcs(img_shape_os, trj::CuArray{T,3}; mask=CUDA.ones(Bool, size(trj)[2:end])) where {T}
    @assert all([i .== nextprod((2, 3, 5), i) for i ∈ img_shape_os]) "img_shape_os has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    backend = CUDABackend()
    p = PlanNUFFT(Complex{T}, img_shape_os; σ=1, kernel=GaussianKernel(), backend=backend) # default is without fftshift
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj[:, mask]))

    S = CUDA.ones(Complex{T}, size(p.points[1]))
    NonuniformFFTs.spread_from_points!(p.backend, NUFFTCallbacks().nonuniform, p.point_transform_fold, p.blocks, p.kernels, p.kernel_evalmode, p.data.us, p.points, (S,))
    kmask_indcs = findall(vec(p.data.us[1] .!= 0))
    return kmask_indcs
end

# kernel is assumed to be real-valued (method only works with real basis U)
function calculateToeplitzKernelBasis(img_shape_os, trj::CuArray{T,3}, U::CuArray{T}; mask=CUDA.ones(Bool, size(trj)[2:end]), verbose=false) where {T <: Real}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj; mask)

    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    nsamp_t = cu(sum(mask, dims=1))
    Ncoeff = size(U, 2)

    # Allocate kernel arrays, write Λ as packed storage arrays (https://www.netlib.org/lapack/lug/node123.html)
    λ  = CuArray{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    Λ  = CuArray{T}(undef, Int(Ncoeff*(Ncoeff+1)/2), length(kmask_indcs)) # requires basis U to be real

    S = CuArray{Complex{T}}(undef, sum(nsamp_t))

    # Prep FFT and NUFFT plans
    fftplan  = plan_fft(λ)
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200)) # use plan specific to real inputs
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj[:, mask]))

    # Kernel helpers
    cumsum_nsamp = cumsum(nsamp_t[1:end-1]) |> x -> cat(CUDA.zeros(eltype(x), 1), x; dims=1);

    # Params for kernel_uprod!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_y = min(max_threads, maximum(nsamp_t))
    threads_x = min(max_threads ÷ threads_y, length(nsamp_t))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (length(nsamp_t), maximum(nsamp_t)) ./ threads)

    # Params for kernel_sort!
    threads_sort = min(max_threads, length(kmask_indcs))
    blocks_sort = ceil.(Int, length(kmask_indcs) ./ threads_sort)

    for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @cuda threads=threads blocks=blocks kernel_uprod!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)

                exec_type1!(λ2, nfftplan, vec(S)) # type 1: non-uniform points to uniform grid
                mul!(λ, fftplan, λ2)

                @cuda threads=threads_sort blocks=blocks_sort kernel_sort!(Λ, λ, kmask_indcs, ic1, ic2)
            end
            verbose && (CUDA.synchronize(); println("ic = ($ic1, $ic2): t = $t s"); flush(stdout))
        end
    end
    return Λ, kmask_indcs
end

function LinearAlgebra.mul!(x::CuArray, S::MRFingerprintingRecon._NFFTNormalOp, b, α, β)
    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    idx = CartesianIndices(S.shape)

    for cmap ∈ S.cmaps
        fill!(S.kL1, 0)
        fill!(S.kL2, 0)
        S.kL1[idx, :] .= cmap .* b
        S.fftplan * S.kL1

        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff)
        @cuda threads=S.threads blocks=S.blocks kernel_mul!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs, S.ind_lookup)

        S.ifftplan * S.kL2
        @views xr .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end
    return x
end

function kernel_uprod!(S, U, nsamp_t, cumsum_nsamp, ic1, ic2)
    it = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ik = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Multiply signal vector by basis, accounting for varying number of samples per time frame
    if it <= length(nsamp_t)
        Uprod = U[it, ic1] * U[it, ic2]
        if ik <= nsamp_t[it]
            S[cumsum_nsamp[it] + ik] = Uprod
            return
        end
    end
end

# Place non-zero elements of kernel as real values in Λ
function kernel_sort!(Λ, λ, kmask_indcs, ic1, ic2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # Packed storage of Λ by columns
    ind_packed = ic1 + ic2 * (ic2-1) ÷ 2
    if i <= length(kmask_indcs)
        Λ[ind_packed, i] = real(λ[kmask_indcs[i]]) # assumes basis U is real
    end
    return
end

function kernel_mul!(kL2_rs, Λ, kL1_rs, kmask_indcs, ind_lookup)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= length(kmask_indcs) && j <= size(kL2_rs, 2)
        ind = kmask_indcs[i]
        acc = zero(eltype(kL2_rs))

        @inbounds for k ∈ axes(ind_lookup, 2)
            ind_packed = ind_lookup[j, k]
            acc += Λ[ind_packed, i] * kL1_rs[ind, k]
        end

        kL2_rs[ind, j] = acc
    end
    return
end

# Wrapper for 4D data arrays
function MRFingerprintingRecon.NFFTNormalOp(img_shape, trj::CuArray{T,4}, U::CuArray{T}; mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...) where {T}
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return MRFingerprintingRecon.NFFTNormalOp(img_shape, trj, U; kwargs..., mask)
end