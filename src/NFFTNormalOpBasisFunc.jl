
struct NFFTNormalOpBasisFunc{S,D,T,E,F}
    shape::S
    Ncoeff::Int
    weights::D
    fftplan::E
    ifftplan::F
    λ::Array{Complex{T},3}
    kL1::Array{Complex{T}}
    kL2::Array{Complex{T}}
    cmaps
end

function NFFTNormalOpBasisFunc(img_shape, tr::Vector{Matrix{T}}, U::Matrix{Complex{T}}, cmaps = (1,); verbose=false) where {T}
    img_shape_os = 2 .* img_shape
    Ncoeff = size(U, 2)

    kL1 = Array{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = similar(kL1)

    xtmp = Array{Complex{T}}(undef, img_shape_os)

    FFTW.set_num_threads(Threads.nthreads())
    fftplan = plan_fft(xtmp; flags = FFTW.MEASURE)
    ifftplan = plan_ifft(xtmp; flags = FFTW.MEASURE)
    nfftplan = plan_nfft(tr[1], img_shape_os, m=4, σ=2; fftflags = FFTW.MEASURE)
    verbose && println("Planned FFTs")

    λ = Array{Complex{T}}(undef, img_shape_os)
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, prod(img_shape_os))
    Λ .= 0
    verbose && println("Λ initialized")

    for i ∈ eachindex(tr)
        verbose && println(string("Time frame ", i))
        verbose && @time calculateToeplitzKernel!(λ, nfftplan, tr[i], fftplan)
        !verbose && calculateToeplitzKernel!(λ, nfftplan, tr[i], fftplan)

        @views U2 = conj.(U[i, :]) * transpose(U[i, :])
        # @time begin
            @batch for j ∈ eachindex(λ)
                @simd for iu ∈ CartesianIndices(U2)
                    @inbounds Λ[iu, j] += U2[iu] * λ[j]
                end
            end
        # end
    end

    return NFFTNormalOpBasisFunc(img_shape, Ncoeff, I, fftplan, ifftplan, Λ, kL1, kL2, cmaps)
end


function LinearAlgebra.mul!(x::Vector{T}, S::NFFTNormalOpBasisFunc, b) where {T}
    idx = CartesianIndices(S.shape)
    idxos = CartesianIndices(2 .* S.shape)
    Ncoils = length(S.cmaps)

    b = reshape(b, S.shape..., S.Ncoeff)
    xL = @view S.kL2[idxos, 1]
    fill!(x, zero(T))

    for icoil ∈ 1:Ncoils
        fill!(xL, zero(T))

        @inbounds for i = 1:S.Ncoeff
            @views xL[idx] .= S.cmaps[icoil] .* b[idx, i]
            @views mul!(S.kL1[idxos, i], S.fftplan, xL)
        end

        kin = reshape(S.kL1, :, S.Ncoeff)
        kout = reshape(S.kL2, :, S.Ncoeff)
        @batch for i ∈ eachindex(view(S.λ, 1, 1, :))
            @views @inbounds mul!(kout[i, :], S.λ[:, :, i], kin[i, :])
        end

        @batch for i = 1:S.Ncoeff
            @views mul!(S.kL1[idxos, i], S.ifftplan, S.kL2[idxos, i])
        end

        @views x .+= vec(conj.(S.cmaps[icoil]) .* S.kL1[idx, :])
    end
    return x
end


Base.:*(S::NFFTNormalOpBasisFunc, b::AbstractVector) = mul!(similar(b), S, b)
Base.size(S::NFFTNormalOpBasisFunc) = S.shape
Base.size(S::NFFTNormalOpBasisFunc, dim) = S.shape[dim]
Base.eltype(::Type{NFFTNormalOpBasisFunc{S,D,T}}) where {S,D,T} = T