## ##########################################################################
# NFFTNormalOp
#############################################################################

"""
    NFFTNormalOp(img_shape, trj, U; cmaps, verbose, num_fft_threads)
    NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps)

Create normal operator of NFFT operator.
Differentiate between functions exploiting a pre-calculated Toeplitz kernel basis `Λ` and the function which calculates Λ based on a passed trajectory `trj`.

# Arguments
- `img_shape::Tuple{Int}`: Image dimensions
- `traj::AbstractVector{<:AbstractMatrix{T}}`: Trajectory
- `U::AbstractMatrix{Tc}`: Basis coefficients of subspace
- `cmaps::AbstractVector{Matrix{ComplexF32}}`=`[ones(T, img_shape)]`: Coil sensitivities
- `Λ::Array{Complex{T},3}`: Toeplitz kernel basis
- `kmask_indcs::Vector{Int}`: Sampling indices of Toeplitz mask
- `verbose::Boolean`=`false`: Verbose level
- `num_fft_threads::Int`=`round(Int, Threads.nthreads()/size(U, 2))` or `round(Int, Threads.nthreads()/size(Λ, 1))`: Number of threads for FFT
"""
function NFFTNormalOp(
    img_shape,
    trj::AbstractVector{<:AbstractMatrix{T}},
    U::AbstractArray{Tc};
    cmaps=[ones(T, img_shape)],
    verbose = false,
    num_fft_threads = round(Int, Threads.nthreads()/size(U, 2)),
    ) where {T, Tc <: Union{T, Complex{T}}}

    Λ, kmask_indcs = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; verbose=verbose)

    return NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps, num_fft_threads=num_fft_threads)
end

function NFFTNormalOp(
    img_shape,
    trj::CuArray{T},
    trj_l,
    U::CuArray{Tc};
    cmaps = [CUDA.ones(T, img_shape)],
    verbose = false
    ) where {T, Tc <:Union{T, Complex{T}}}

    Λ, kmask_indcs = calculateToeplitzKernelBasis(2 .* img_shape, trj, trj_l, U; verbose=verbose)

    return NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps)
end

function NFFTNormalOp(
    img_shape,
    Λ::Array{Tc,3},
    kmask_indcs;
    cmaps=[ones(T, img_shape)],
    num_fft_threads = round(Int, Threads.nthreads()/size(Λ, 1))
    ) where {T, Tc <:Union{T, Complex{T}}}

    @assert length(kmask_indcs) == size(Λ,3) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))

    Ncoeff = size(Λ, 1)
    img_shape_os = 2 .* img_shape
    kL1 = Array{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = similar(kL1)

    ktmp = @view kL1[CartesianIndices(img_shape_os),1]

    fftplan  = plan_fft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)
    ifftplan = plan_ifft!(ktmp; flags = FFTW.MEASURE, num_threads=num_fft_threads)

    A = _NFFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps)

	return LinearOperator(
        Complex{T},
        prod(A.shape) * A.Ncoeff,
        prod(A.shape) * A.Ncoeff,
        true,
        true,
        (res, x, α, β) -> mul!(res, A, x, α, β),
        nothing,
        (res, x, α, β) -> mul!(res, A, x, α, β),
    )
end

function NFFTNormalOp(
    img_shape,
    Λ::CuArray{T},
    kmask_indcs;
    cmaps=[CuArray(ones(T, img_shape))]
    ) where {T}

    @assert length(kmask_indcs) == size(Λ, length(size(Λ))) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))

    packed_length = size(Λ, 1) # derive Ncoeff from length of packed axis using quadratic eqn
    Ncoeff = Int(0.5 * (-1 + sqrt(8 * packed_length + 1)))

    img_shape_os = 2 .* img_shape
    kL1 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = CuArray{Complex{T}}(undef, img_shape_os..., Ncoeff)

    fftplan  = plan_fft!(kL1, Vector(1:length(img_shape_os)))
    ifftplan = plan_ifft!(kL2, Vector(1:length(img_shape_os)))

    A = _NFFTNormalOp(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps)

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
struct _NFFTNormalOp{S,E,F,G,H,I,J,K}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::G
    kmask_indcs::H
    kL1::I
    kL2::J
    cmaps::K
end

function calculate_kmask_indcs(img_shape_os, trj::AbstractVector{<:AbstractMatrix{T}}) where T
    @assert all([i .== nextprod((2, 3, 5), i) for i ∈ img_shape_os]) "img_shape_os has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    backend = CPU()
    p = PlanNUFFT(Complex{T}, img_shape_os; σ=1, kernel=GaussianKernel(), backend=backend) # default is without fftshift
    set_points!(p, NonuniformFFTs._transform_point_convention.(reduce(hcat, trj)))

    S = ones(Complex{T}, size(p.points[1]))
    NonuniformFFTs.spread_from_points!(p.backend, NUFFTCallbacks().nonuniform, p.point_transform_fold, p.blocks, p.kernels, p.kernel_evalmode, p.data.us, p.points, (S,))
    kmask_indcs = findall(vec(p.data.us[1] .!= 0))
    return kmask_indcs
end

function calculate_kmask_indcs(img_shape_os, trj::CuArray{T}) where T
    @assert all([i .== nextprod((2, 3, 5), i) for i ∈ img_shape_os]) "img_shape_os has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    backend = CUDABackend()
    p = PlanNUFFT(Complex{T}, img_shape_os; σ=1, kernel=GaussianKernel(), backend=backend) # default is without fftshift
    set_points!(p, NonuniformFFTs._transform_point_convention.(trj))

    S = CUDA.ones(Complex{T}, size(p.points[1]))
    NonuniformFFTs.spread_from_points!(p.backend, NUFFTCallbacks().nonuniform, p.point_transform_fold, p.blocks, p.kernels, p.kernel_evalmode, p.data.us, p.points, (S,))
    kmask_indcs = findall(vec(p.data.us[1] .!= 0))
    return kmask_indcs
end

function calculateToeplitzKernelBasis(img_shape_os, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractArray{Tc}; verbose = false) where {T, Tc <: Complex{T}}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)

    λ  = Array{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    Λ  = Array{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs))

    trj_idx = cumsum([size(trj[it],2) for it in eachindex(trj)])
    S  = Vector{Complex{T}}(undef, trj_idx[end])

    fftplan  = plan_fft(λ; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os) # default is without fftshift
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reduce(hcat, trj)))

    # Evaluating only the upper triangular matrix assumes that the PSF from the rightmost voxel to the leftmost voxel is the adjoint of the PSF in the opposite direction.
    # For the outmost voxel, this is not correct, but the resulting images are virtually identical in our test cases.
    # To avoid this error, remove the `if ic2 >= ic1` and the `Λ[ic2,ic1,it] = conj.(λ[kmask_indcs[it]])` statements at the cost of computation time.
    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @simd for it ∈ axes(U,1)
                    idx1 = (it == 1) ? 1 : trj_idx[it-1] + 1
                    idx2 = trj_idx[it]
                    @inbounds S[idx1:idx2] .= conj(U[it,ic1]) * U[it,ic2]
                end

                exec_type1!(λ2, nfftplan, vec(S))
                mul!(λ, fftplan, λ2)

                Threads.@threads for it ∈ eachindex(kmask_indcs)
                    @inbounds Λ[ic2,ic1,it] = conj.(λ[kmask_indcs[it]])
                    @inbounds Λ[ic1,ic2,it] =       λ[kmask_indcs[it]]
                end
            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end
    return Λ, kmask_indcs
end

function calculateToeplitzKernelBasis(img_shape_os, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractArray{T}; verbose=false) where {T <: Real}
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)
    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)

    λ  = Array{T}(undef, img_shape_os)
    λ2 = Array{Complex{T}}(undef, img_shape_os[1] ÷ 2 + 1, Base.tail(img_shape_os)...)
    Λ  = Array{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs))

    trj_idx = cumsum([size(trj[it],2) for it in eachindex(trj)])
    S  = Vector{T}(undef, trj_idx[end])

    brfftplan = plan_brfft(λ2, img_shape_os[1]; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = PlanNUFFT(T, img_shape_os) # use plan specific to real inputs
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(reduce(hcat, trj)))

    # Evaluating only the upper triangular matrix assumes that the PSF from the rightmost voxel to the leftmost voxel is the adjoint of the PSF in the opposite direction.
    # For the outmost voxel, this is not correct, but the resulting images are virtually identical in our test cases.
    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @simd for it ∈ axes(U,1)
                    idx1 = (it == 1) ? 1 : trj_idx[it-1] + 1
                    idx2 = trj_idx[it]
                    @inbounds S[idx1:idx2] .= U[it,ic1] * U[it,ic2]
                end

                exec_type1!(λ2, nfftplan, vec(S))
                λ2 .= conj.(λ2) # conjugate input to flip the sign of the exponential in brfft
                mul!(λ, brfftplan, λ2)

                Threads.@threads for it ∈ eachindex(kmask_indcs)
                    @inbounds Λ[ic2,ic1,it] = λ[kmask_indcs[it]]
                    @inbounds Λ[ic1,ic2,it] = λ[kmask_indcs[it]]
                end
            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end
    return Λ, kmask_indcs
end

function calculateToeplitzKernelBasis(img_shape_os, trj::CuArray{T}, trj_l, U::CuArray{Tc}; verbose = false) where {T, Tc <: Union{T, Complex{T}}}

    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)

    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)
    Nt = size(U,1)

    # Allocate kernel arrays, write Λ as packed storage arrays
    λ  = CuArray{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    Λ  = CuArray{T}(undef, Int(Ncoeff*(Ncoeff+1)/2), length(kmask_indcs))

    # trj_l = [size(trj[it],2) for it in eachindex(trj)]
    S = CuArray{Complex{T}}(undef, sum(trj_l))

    # Prep plans
    fftplan  = plan_fft(λ)
    nfftplan = PlanNUFFT(Complex{T}, img_shape_os; backend=CUDABackend(), gpu_method=:shared_memory, gpu_batch_size = Val(200)) # use plan specific to real inputs
    set_points!(nfftplan, NonuniformFFTs._transform_point_convention.(trj))

    # Kernel helpers
    Uc = conj(U)
    trj_c = CuArray([0; cumsum(trj_l[1:end-1])])
    trj_l = CuArray(trj_l)

    # Params for kernel_uprod!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_y = min(max_threads, maximum(trj_l))
    threads_x = min(max_threads ÷ threads_y, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (Nt, maximum(trj_l)) ./ threads)

    # Params for kernel_sort!
    threads_sort = min(max_threads, length(kmask_indcs))
    blocks_sort = ceil.(Int, length(kmask_indcs) ./ threads_sort)

    for ic2 ∈ 1:Ncoeff, ic1 ∈ 1:Ncoeff
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @cuda threads=threads blocks=blocks kernel_uprod!(S, Uc, U, trj_l, trj_c, Nt, ic1, ic2)

                exec_type1!(λ2, nfftplan, vec(S))
                mul!(λ, fftplan, λ2)

                @cuda threads=threads_sort blocks=blocks_sort kernel_sort!(Λ, λ, kmask_indcs, ic1, ic2)
            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end
    return Λ, kmask_indcs
end

function LinearAlgebra.mul!(x::AbstractVector{T}, S::_NFFTNormalOp, b, α, β) where {T}

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
                @views S.fftplan * S.kL1[idxos, i]
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
                @views S.ifftplan * S.kL2[idxos, i]
                @views xr[idx,i] .+= α .* conj.(cmap) .* S.kL2[idx,i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end

function LinearAlgebra.mul!(x::CuArray, S::_NFFTNormalOp, b, α, β)

    b = reshape(b, S.shape..., S.Ncoeff)
    if β == 0
        x .= 0
    else
        x .*= β
    end
    xr = reshape(x, S.shape..., S.Ncoeff)

    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)

    opt_threads = 768 # threads optimized for an NVIDIA A100 and our usual data arrays using CUDA.launch_configuration()
    max_threads = min(opt_threads, attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    threads_x = min(max_threads, length(S.kmask_indcs))
    threads_y = min(max_threads ÷ threads_x, S.Ncoeff)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (length(S.kmask_indcs), S.Ncoeff) ./ threads)
    ind_lookup = CuArray([j<k ? j+k*(k-1)÷2 : k+j*(j-1)÷2 for j ∈ 1:S.Ncoeff, k ∈ 1:S.Ncoeff])

    for cmap ∈ S.cmaps
        S.kL1[idxos, :] .= 0
        @views S.kL1[idx, :] .= cmap .* b[idx, :]
        S.fftplan * S.kL1

        kL1_rs = reshape(S.kL1, :, S.Ncoeff)
        kL2_rs = reshape(S.kL2, :, S.Ncoeff) .= 0
        @cuda threads=threads blocks=blocks kernel_mul!(kL2_rs, S.Λ, kL1_rs, S.kmask_indcs, S.Ncoeff, ind_lookup)

        S.ifftplan * S.kL2
        @views xr[idx, :] .+= α .* conj.(cmap) .* S.kL2[idx, :]
    end
    return x
end

function kernel_uprod!(S, Uc, U, trj_l, trj_c, Nt, ic1, ic2)

    it = (blockIdx().x - 1) * blockDim().x + threadIdx().x # time index
    ik = (blockIdx().y - 1) * blockDim().y + threadIdx().y # sample index

    if it <= Nt
        Uprod = Uc[it, ic1] * U[it, ic2]
        if ik <= trj_l[it]
            S[trj_c[it] + ik] = Uprod
            return
        end
    end
end

function kernel_sort!(Λ, λ, kmask_indcs, ic1, ic2)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # Packed storage of Λ by columns
    ind_pack = ic1 + ic2 * (ic2-1) ÷ 2
    if i <= length(kmask_indcs)
        Λ[ind_pack, i] = real(λ[kmask_indcs[i]])
    end
    return
end

function kernel_mul!(kL2_rs, Λ, kL1_rs, kmask_indcs, Ncoeff, ind_lookup)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # Parallelize across kmask and row index j
    if i <= length(kmask_indcs) && j <= size(kL2_rs, 2)

        ind = kmask_indcs[i]
        for k in 1:Ncoeff
            ind_packed = ind_lookup[j,k]
            kL2_rs[ind, j] += Λ[ind_packed, i] * kL1_rs[ind, k]
        end
    end
    return
end
