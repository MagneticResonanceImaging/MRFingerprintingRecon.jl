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

#CPU
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

# GPU
function NFFTNormalOp(
    img_shape, 
    trj::AbstractVector{<:CuArray{T}},
    U::CuArray{Tc};
    cmaps = [CuArray(ones(T, img_shape))],
    verbose = false
    ) where {T, Tc <:Union{T, Complex{T}}}

    Λ, kmask_indcs = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; verbose=verbose)

    return NFFTNormalOp(img_shape, Λ, kmask_indcs; cmaps=cmaps)
end

#CPU
function NFFTNormalOp(
    img_shape,
    Λ::Array{Complex{T},3},
    kmask_indcs;
    cmaps=[ones(T, img_shape)],
    num_fft_threads = round(Int, Threads.nthreads()/size(Λ, 1))
    ) where {T}

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

#GPU
function NFFTNormalOp(
    img_shape,
    Λ::CuArray{Complex{T}, 3},
    kmask_indcs;
    cmaps=[CuArray(ones(T, img_shape))]
    ) where {T}

    @assert length(kmask_indcs) == size(Λ,3) # ensure that kmask is not out of bound as we use `@inbounds` in `mul!`
    @assert all(kmask_indcs .> 0)
    @assert all(kmask_indcs .<= prod(2 .* img_shape))  
    
    Ncoeff = size(Λ, 1)
    img_shape_os = 2 .* img_shape
    kL1 = CuArray{Complex{T}}(undef, img_shape_os)
    kL2 = nothing

    ktmp = kL1[CartesianIndices(size(kL1))]

    fftplan  = plan_fft!(ktmp, Vector(1:length(img_shape_os)))
    ifftplan = plan_ifft!(ktmp, Vector(1:length(img_shape_os)))

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


## ##########################################################################
# Internal use
#############################################################################

# struct _NFFTNormalOp{S,T,N,E,F,G}
#     shape::S
#     Ncoeff::Int
#     fftplan::E
#     ifftplan::F
#     Λ::Array{Complex{T},3}
#     kmask_indcs::Vector{Int}
#     kL1::Array{Complex{T},N}
#     kL2::Array{Complex{T},N}
#     cmaps::G
# end

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
    nfftplan = plan_nfft(reduce(hcat, trj), img_shape_os; precompute = POLYNOMIAL, blocking = false, fftflags = FFTW.MEASURE, m=5, σ=1)

    convolve_transpose!(nfftplan, ones(Complex{T}, size(nfftplan)[1]), nfftplan.tmpVec)
    kmask = (nfftplan.tmpVec .!= 0)
    kmask_indcs = findall(vec(kmask))
    return kmask_indcs
end

function calculate_kmask_indcs(img_shape_os, trj::AbstractVector{<:CuArray{T}}) where T
    shape = Int.(img_shape_os ./ 2)
    trj_v = reduce(hcat, trj)
    nfftplan = PlanNUFFT(trj_v, shape; backend=CUDABackend())
    (; backend, points, kernels, data, blocks,) = nfftplan
    (; us,) = data
    vp = (CuArray(ones(Complex{T}, size(trj_v, 2))),)
    NonuniformFFTs.spread_from_points!(backend, blocks, kernels, us, points, vp)
    kmask = (us[1] .!= 0)
    kmask_indcs = findall(vec(kmask))
    return kmask_indcs
end

function calculateToeplitzKernelBasis(img_shape_os, trj::AbstractVector{<:AbstractMatrix{T}}, U::AbstractArray{Tc}; verbose = false) where {T, Tc <: Union{T, Complex{T}}}

    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)

    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bounds
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)

    λ  = Array{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    Λ  = Array{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs))

    trj_l = [size(trj[it],2) for it in eachindex(trj)]
    S  = Vector{Complex{T}}(undef, sum(trj_l))

    fftplan  = plan_fft(λ; flags = FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = plan_nfft(reduce(hcat, trj), img_shape_os; precompute = TENSOR, blocking = true, fftflags = FFTW.MEASURE, m=5, σ=2)

    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @simd for it ∈ axes(U,1)
                    idx1 = sum(trj_l[1:it-1]) + 1
                    idx2 = sum(trj_l[1:it]) 
                    @inbounds S[idx1:idx2] .= conj(U[it,ic1]) * U[it,ic2] 
                end

                mul!(λ, adjoint(nfftplan), vec(S)) 
                fftshift!(λ2, λ) 
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

function calculateToeplitzKernelBasis(img_shape_os, trj::AbstractVector{<:CuArray{T}}, U::CuArray{Tc}; verbose = false) where {T, Tc <: Union{T, Complex{T}}}
    
    kmask_indcs = calculate_kmask_indcs(img_shape_os, trj)

    @assert all(kmask_indcs .> 0) # ensure that kmask is not out of bound
    @assert all(kmask_indcs .<= prod(img_shape_os))

    Ncoeff = size(U, 2)
    Nt = size(U,1)

    # Allocate kernel arrays
    λ  = CuArray{Complex{T}}(undef, img_shape_os) # upper triangle for (ic1, ic2)
    λ2 = similar(λ)
    Λ  = CuArray{Complex{T}}(undef, Ncoeff, Ncoeff, length(kmask_indcs)) # Pack
  
    trj_l = [size(trj[it],2) for it in eachindex(trj)]
    S = CuArray{Complex{T}}(undef, sum(trj_l)) 

    # Prep plans
    fftplan  = plan_fft(λ)
    nfftplan = PlanNUFFT(reduce(hcat, trj), img_shape_os; backend=CUDABackend())

    # Kernel helpers
    Uc = conj(U)
    trj_c = CuArray([0; cumsum(trj_l[1:end-1])])
    trj_l = CuArray(trj_l)

    # @cuda arguments
    # 1. kernel_mul_tpk!
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) 
    threads_y = min(max_threads, maximum(trj_l))
    threads_x = min(max_threads ÷ threads_y, Nt)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (Nt, maximum(trj_l)) ./ threads)

    # 2. kernel_sort!
    threads_sort = min(max_threads, length(kmask_indcs))
    blocks_sort = ceil.(Int, length(kmask_indcs) ./ threads_sort)

    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
         
                @cuda threads=threads blocks=blocks kernel_mul_tpk!(S, Uc, U, trj_l, trj_c, Nt, ic1, ic2)

                mul!(λ, adjoint(nfftplan), vec(S)) 
                fftshift!(λ2, λ) 
                mul!(λ, fftplan, λ2) 
                λ2 .= conj.(λ)

                @cuda threads=threads_sort blocks=blocks_sort kernel_sort!(Λ, λ, λ2, kmask_indcs, ic1, ic2)

            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end

    return Λ, kmask_indcs 
end


#CPU
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

function kernel_mul_tpk!(S, Uc, U, trj_l, trj_c, Nt, ic1, ic2)

    # Get current time index
    it = (blockIdx().x - 1) * blockDim().x + threadIdx().x # time index
    ik = (blockIdx().y - 1) * blockDim().y + threadIdx().y # sample index

    # Fill samples with product of basis funcs
    if it <= Nt
        # Precompute product once
        Uprod = Uc[it, ic1] * U[it, ic2]
        if ik <= trj_l[it] 
            S[trj_c[it] + ik] = Uprod 
            return
        end
    end
end

function kernel_sort!(Λ, λ, λ2, kmask_indcs, ic1, ic2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    if i <= length(kmask_indcs)
        Λ[ic2,ic1,i] = λ2[kmask_indcs[i]]
        Λ[ic1,ic2,i] =  λ[kmask_indcs[i]]
    end
    return
end

#GPU
# function LinearAlgebra.mul!(x::CuArray, S::_NFFTNormalOp, b, α, β)

#     b = reshape(b, S.shape..., S.Ncoeff)
#     if β == 0
#         x .= 0
#     else
#         x .*= β
#     end
#     xr = reshape(x, S.shape..., S.Ncoeff)

#     idx = CartesianIndices(S.shape)
#     idxos = CartesianIndices(2 .* S.shape)

#     max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) 
#     threads = min(max_threads, length(S.kmask_indcs))
#     blocks = ceil(Int, length(S.kmask_indcs) / threads)

#     for cmap ∈ S.cmaps
#         for ic1 ∈ axes(S.Λ, 1), ic2 ∈ axes(S.Λ, 2)
#             S.kL1[idxos] .= 0
#             @views S.kL1[idx] .= cmap .* b[idx, ic2]
#             S.fftplan * S.kL1
            
#             kL1_rs = vec(S.kL1)

#             @cuda threads=threads blocks=blocks kernel_mul!(kL1_rs, S.Λ, S.kmask_indcs, ic1, ic2)

#             S.ifftplan * S.kL1
#             @views xr[idx, ic1] .+= α .* conj.(cmap) .* S.kL1[idx]
#         end
#     end
#     return x
# end

# GPU
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

    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) 
    threads = min(max_threads, length(S.kmask_indcs))
    blocks = ceil(Int, length(S.kmask_indcs) / threads)

    for cmap ∈ S.cmaps
        for ic1 ∈ axes(S.Λ, 1), ic2 ∈ axes(S.Λ, 2) 

            S.kL1[idxos] .= 0
            @views S.kL1[idx] .= cmap .* b[idx, ic2] # image shape
            S.fftplan * S.kL1 # image shape out
            kL1_rs = vec(S.kL1) 

            # Pack
            @cuda threads=threads blocks=blocks kernel_mula!(kL1_rs, S.Λ, S.kmask_indcs, ic1, ic2)

            S.ifftplan * S.kL1
            @views xr[idx, ic1] .+= α .* conj.(cmap) .* S.kL1[idx]
        end
    end
    return x
end

function kernel_mula!(kL1_rs, Λ, kmask_indcs, ic1, ic2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(kmask_indcs)
        ind = kmask_indcs[i]
        if ic2 >= ic1
            kL1_rs[ind] *= Λ[ic1, ic2, i] # Pack
        else
            kL1_rs[ind] *= conj(Λ[ic2, ic1, i]) # Pack
        end
    end
    return 
end
