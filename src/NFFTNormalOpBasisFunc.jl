function calculateToeplitzKernelBasis(img_shape_os, trj::Vector{Matrix{T}}, U::Matrix{Complex{T}}; verbose = false) where {T}

    Ncoeff = size(U, 2)
    Nt = size(U,1)
    Nk = size(trj[1],2)

    λ  = Array{Complex{T}}(undef, img_shape_os)
    λ2 = similar(λ)
    λ3 = similar(λ)
    Λ  = Array{Complex{T}}(undef, Ncoeff, Ncoeff, prod(img_shape_os))
    S  = Array{Complex{T}}(undef, Nk, Nt)

    fftplan  = plan_fft(λ; flags = FFTW.MEASURE, num_threads=Threads.nthreads())
    nfftplan = plan_nfft(reduce(hcat, trj), img_shape_os; precompute = TENSOR, blocking = true, fftflags = FFTW.MEASURE, m=5, σ=2)

    for ic2 ∈ axes(Λ, 2), ic1 ∈ axes(Λ, 1)
        if ic2 >= ic1 # eval. only upper triangular matrix
            t = @elapsed begin
                @simd for it ∈ axes(U,1)
                    @inbounds S[:,it] .= conj(U[it,ic1]) * U[it,ic2]
                end

                mul!(λ, adjoint(nfftplan), vec(S))
                fftshift!(λ2, λ)
                mul!(λ, fftplan, λ2)
                λ2 .= conj.(λ2)
                mul!(λ3, fftplan, λ2)

                Threads.@threads for it ∈ eachindex(λ)
                    @inbounds Λ[ic2,ic1,it] = λ3[it]
                    @inbounds Λ[ic1,ic2,it] = λ[it]
                end
            end
            verbose && println("ic = ($ic1, $ic2): t = $t s"); flush(stdout)
        end
    end

    return Λ
end

## ##########################################################################
# NFFTNormalOpBasisFunc
#############################################################################
struct NFFTNormalOpBasisFunc{S,T,E,F,G}
    shape::S
    Ncoeff::Int
    fftplan::E
    ifftplan::F
    Λ::Array{Complex{T},3}
    kL1::Array{Complex{T}}
    kL2::Array{Complex{T}}
    cmaps::G
end

function NFFTNormalOpBasisFunc(
    img_shape,
    trj::Vector{Matrix{T}},
    U::Matrix{Complex{T}};
    cmaps = (1,),
    verbose = false,
    Λ = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; verbose = verbose),
    ) where {T}

    Ncoeff = size(U, 2)
    img_shape_os = 2 .* img_shape
    kL1 = Array{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = similar(kL1)

    ktmp = @view kL1[CartesianIndices(img_shape_os),1]
    fftplan  = plan_fft!( ktmp; flags = FFTW.MEASURE, num_threads=round(Int, Threads.nthreads()/Ncoeff))
    ifftplan = plan_ifft!(ktmp; flags = FFTW.MEASURE, num_threads=round(Int, Threads.nthreads()/Ncoeff))
    return NFFTNormalOpBasisFunc(img_shape, Ncoeff, fftplan, ifftplan, Λ, kL1, kL2, cmaps)
end


function LinearAlgebra.mul!(x::Vector{T}, S::NFFTNormalOpBasisFunc, b, α, β) where {T}
    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)
    Ncoils = length(S.cmaps)

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
        for icoil ∈ 1:Ncoils
            Threads.@threads for i ∈ 1:S.Ncoeff
                S.kL1[idxos, i] .= zero(T)
                @views S.kL1[idx, i] .= S.cmaps[icoil] .* b[idx, i]
                @views S.fftplan * S.kL1[idxos, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i ∈ axes(S.Λ, 3)
                @views @inbounds mul!(kL2_rs[i, :], S.Λ[:, :, i], kL1_rs[i, :])
            end

            Threads.@threads for i ∈ 1:S.Ncoeff
                @views S.ifftplan * S.kL2[idxos, i]
                @views xr[idx,i] .+= α .* conj.(S.cmaps[icoil]) .* S.kL2[idx,i]
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end
    return x
end


Base.:*(S::NFFTNormalOpBasisFunc, b::AbstractVector) = mul!(similar(b), S, b)
Base.size(S::NFFTNormalOpBasisFunc) = S.shape
Base.size(S::NFFTNormalOpBasisFunc, dim) = S.shape[dim]
Base.eltype(::Type{NFFTNormalOpBasisFunc{S,D,T}}) where {S,D,T} = T


## ##########################################################################
# LinearOperator of NFFTNormalOpBasisFunc
#############################################################################
function NFFTNormalOpBasisFuncLO(A::NFFTNormalOpBasisFunc{S,T,E,F,G}) where {S,T,E,F,G}
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

function NFFTNormalOpBasisFuncLO(
    img_shape,
    trj::Vector{Matrix{T}},
    U::Matrix{Complex{T}};
    cmaps = (1,),
    verbose = false,
    Λ = calculateToeplitzKernelBasis(2 .* img_shape, trj, U; verbose = verbose),
    ) where {T}

    S = NFFTNormalOpBasisFunc(img_shape, trj, U; cmaps = cmaps, Λ = Λ)
    return NFFTNormalOpBasisFuncLO(S)
end