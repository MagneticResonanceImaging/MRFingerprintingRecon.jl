function calculateKernelBasis(img_shape, D::AbstractArray{G}, U::Matrix{Complex{T}}; verbose = false) where {G,T}

    Ncoeff = size(U,2)
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, img_shape...)
    t = @elapsed begin
        Threads.@threads for i ∈ CartesianIndices(img_shape) # takes 0.5s for 2D
            Λ[:,:,i] .= U' * (D[i,:] .* U) #U' * diagm(D) * U
        end
        Λ .= ifftshift(Λ, 3:(3+length(img_shape)-1)) #could fftshift D first
    end
    verbose && println("Kernel calculation: t = $t s"); flush(stdout)

    return Λ
end

## ##########################################################################
# FFTNormalOpBasisFunc
#############################################################################
struct FFTNormalOpBasisFunc{S,T,N,E,F,G}
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

function FFTNormalOpBasisFunc(
    img_shape,
    U::Matrix{Complex{T}};
    cmaps = (1,),
    verbose = false,
    D::AbstractArray{G} = ones(Int8, img_shape..., size(U,1)),
    Λ = calculateKernelBasis(img_shape, D, U; verbose = verbose),
    ) where {G,T}

    Ncoeff = size(U, 2)
    kL1 = Array{Complex{T}}(undef, img_shape..., Ncoeff)
    kL2 = similar(kL1)

    @views kmask = (Λ[1,1,CartesianIndices(img_shape)] .!= 0)
    kmask_indcs = findall(vec(kmask))
    Λ = reshape(Λ, Ncoeff, Ncoeff, :)
    Λ = Λ[:,:,kmask_indcs]

    ktmp = @view kL1[CartesianIndices(img_shape),1]
    fftplan = plan_fft!( ktmp; flags = FFTW.MEASURE, num_threads=round(Int, Threads.nthreads()/Ncoeff))
    ifftplan = plan_ifft!(ktmp; flags = FFTW.MEASURE, num_threads=round(Int, Threads.nthreads()/Ncoeff))
    return FFTNormalOpBasisFunc(img_shape, Ncoeff, fftplan, ifftplan, Λ, kmask_indcs, kL1, kL2, cmaps)
end

function LinearAlgebra.mul!(x::Vector{T}, S::FFTNormalOpBasisFunc, b, α, β) where {T}
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
                @views xr[idx,i] .+= α .* conj.(cmap) .* S.kL2[idx,i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end

Base.:*(S::FFTNormalOpBasisFunc, b::AbstractVector) = mul!(similar(b), S, b)
Base.size(S::FFTNormalOpBasisFunc) = S.shape
Base.size(S::FFTNormalOpBasisFunc, dim) = S.shape[dim]
Base.eltype(::Type{FFTNormalOpBasisFunc{S,T,N,E,F,G}}) where {S,T,N,E,F,G} = T


## ##########################################################################
# LinearOperator of FFTNormalOpBasisFunc
#############################################################################
function FFTNormalOpBasisFuncLO(A::FFTNormalOpBasisFunc{S,T,N,E,F,G}) where {S,T,N,E,F,G}
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

function FFTNormalOpBasisFuncLO(
    img_shape,
    U::Matrix{Complex{T}};
    cmaps = (1,),
    verbose = false,
    D::AbstractArray{G} = ones(Int8, img_shape..., size(U,1)),
    Λ = calculateKernelBasis(img_shape, D, U; verbose = verbose),
    ) where {G,T}

    S = FFTNormalOpBasisFunc(img_shape, U; cmaps = cmaps, D=D, Λ = Λ)
    return FFTNormalOpBasisFuncLO(S)
end
