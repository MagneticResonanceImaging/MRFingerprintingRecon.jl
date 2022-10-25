function calculateToeplitzKernelBasis(img_shape_os, trj::Vector{Matrix{T}}, U::Matrix{Complex{T}}; verbose = false) where {T}

    Ncoeff = size(U, 2)

    FFTW.set_num_threads(Threads.nthreads())
    fftplan = plan_fft(Array{Complex{T}}(undef, img_shape_os); flags = FFTW.MEASURE)
    nfftplan = plan_nfft(trj[1], img_shape_os; precompute = POLYNOMIAL, blocking = false, fftflags = FFTW.ESTIMATE)

    λ = Array{Complex{T}}(undef, img_shape_os)
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, prod(img_shape_os))
    Λ .= 0

    for i ∈ eachindex(trj)
        t_kernel = @elapsed calculateToeplitzKernel!(λ, nfftplan, trj[i], fftplan)

        @views U2 = conj.(U[i, :]) * transpose(U[i, :])
        t_multiplication = @elapsed begin
            Threads.@threads for j ∈ eachindex(λ)
                @simd for iu ∈ CartesianIndices(U2)
                    @inbounds Λ[iu, j] += U2[iu] * λ[j]
                end
            end
        end

        if verbose
            println("Time frame $i: t_kernel = $t_kernel; t_multiplication = $t_multiplication")
            flush(stdout)
        end
    end

    return Λ
end

############################################################################################
# NFFTNormalOpBasisFunc
############################################################################################
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

    FFTW.set_num_threads(1)
    fftplan  = plan_fft!(Array{Complex{T}}(undef, 2 .* img_shape); flags = FFTW.MEASURE)
    ifftplan = plan_ifft!(Array{Complex{T}}(undef, 2 .* img_shape); flags = FFTW.MEASURE)
    Ncoeff = size(U, 2)
    kL1 = Array{Complex{T}}(undef, (2 .* img_shape)..., Ncoeff)
    kL2 = similar(kL1)

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

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        for icoil ∈ 1:Ncoils
            Threads.@threads for i = 1:S.Ncoeff
                S.kL1[idxos, i] .= 0
                @views S.kL1[idx, i] .= S.cmaps[icoil] .* b[idx, i]
                @views S.fftplan * S.kL1[idxos, i]
            end

            kL1_rs = reshape(S.kL1, :, S.Ncoeff)
            kL2_rs = reshape(S.kL2, :, S.Ncoeff)
            Threads.@threads for i ∈ eachindex(view(S.Λ, 1, 1, :))
                @views @inbounds mul!(kL2_rs[i, :], S.Λ[:, :, i], kL1_rs[i, :])
            end

            Threads.@threads for i = 1:S.Ncoeff
                @views S.ifftplan * S.kL2[idxos, i]
            end

            @views x .+= α .* vec(conj.(S.cmaps[icoil]) .* S.kL2[idx, :])
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


############################################################################################
# LinearOperator of NFFTNormalOpBasisFunc
############################################################################################
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