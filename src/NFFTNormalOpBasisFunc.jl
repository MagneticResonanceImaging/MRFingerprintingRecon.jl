
struct NFFTNormalOpBasisFunc{S,D,T}
    shape::S
    Ncoeff::Int
    weights::D
    fftplan
    ifftplan
    λ::Array{Complex{T},3}
    kL1::Array{Complex{T}}
    kL2::Array{Complex{T}}
end

function NFFTNormalOpBasisFunc(img_shape, tr::Vector{Matrix{T}}, U::Matrix{Complex{T}}) where T
    img_shape_os = 2 .* img_shape
    Ncoeff = size(U,2)

    # U = Complex.(U) # avoids allocations in mul!(Λ[:,:,j], ui, ui', λ[j], 1)

    kL1 = Array{Complex{T}}(undef, img_shape_os..., Ncoeff)
    kL2 = similar(kL1)

    xtmp = Array{Complex{T}}(undef, img_shape_os)
    fftplan  = plan_fft( xtmp; flags=FFTW.MEASURE)
    ifftplan = plan_ifft(xtmp; flags=FFTW.MEASURE)
    nfftplan = plan_nfft(tr[1], img_shape_os, 4, 2; flags=FFTW.MEASURE)
    println("Planned FFTs")

    # Λ = Array{Matrix{Complex{T}}}(undef, img_shape_os)
    # for i ∈ eachindex(Λ)
    #     Λ[i] = zeros(Complex{T}, Ncoeff, Ncoeff)
    # end
    # println("Λ initialized")

    # for i ∈ eachindex(tr)
    #     println(string("Time frame ", i))
    #     # λ = calculateToeplitzKernel(img_shape_os, tr[i], fftplan)
    #     λ = calculateToeplitzKernel_apx(img_shape_os, tr[i], fftplan)
    #     ui = @view U[i,:]
    #     Threads.@threads for j ∈ eachindex(Λ, λ)
    #         # Λ[j] .+= U[i,:] * (λ[j] .* U[i,:]')
    #         mul!(Λ[j], ui, ui', λ[j], 1)
    #     end
    # end

    λ = Array{Complex{T}}(undef, img_shape_os)
    Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, prod(img_shape_os))
    Λ .= 0
    println("Λ initialized")

    # U2 = Array{Complex{T}}(undef, Ncoeff, Ncoeff)
    for i ∈ eachindex(tr)
        println(string("Time frame ", i))
        @time calculateToeplitzKernel!(λ, nfftplan, tr[i], fftplan)
        # λ = calculateToeplitzKernel(img_shape, tr[i], fftplan=fftplan)

        # ui = U[i,:]
        @views U2 = U[i,:] * U[i,:]'
        @time begin
            @batch for j ∈ eachindex(λ)
                # Λ[:,:,j] .+= U2 .* λ[j]
                # @views mul!(Λ[:,:,j], ui, ui', λ[j], true)
                @simd for iu ∈ CartesianIndices(U2)
                    @inbounds Λ[iu,j] += U2[iu] * λ[j]
                end
            end
        end

        # @views mul!(U2, U[i,:], U[i,:]')
        # @inbounds for j ∈ eachindex(λ)
        #     # mul!(view(Λ,:,:,j), U2, true, λ[j], true)
        #     U2 .*= λ[j]
        #     Λ[:,:,j] .+= U2
        #     U2 ./= λ[j]
        # end
    end

    return NFFTNormalOpBasisFunc(img_shape, Ncoeff, I, fftplan, ifftplan, Λ, kL1, kL2)
end

function Base.size(S::NFFTNormalOpBasisFunc)
    return S.shape
end

function Base.size(S::NFFTNormalOpBasisFunc, dim)
    return S.shape[dim]
end

function LinearAlgebra.mul!(x::Vector{T}, S::NFFTNormalOpBasisFunc, b) where T
    idx = CartesianIndices(S.shape)

    # fill!(S.kL2, zero(T))
    # S.kL2[idx,:] .= reshape(b, S.shape..., S.Ncoeff)
    # @batch for i = 1:S.Ncoeff
    #     @views mul!(S.kL1[:,:,:,i], S.fftplan, S.kL2[:,:,:,i])
    # end

    b = reshape(b, S.shape..., S.Ncoeff)
    xL = @view S.kL2[:,:,:,1]
    fill!(xL, zero(T))
    for i = 1:S.Ncoeff
        @views xL[idx] .= b[:,:,:,i]
        @views mul!(S.kL1[:,:,:,i], S.fftplan, xL)
    end

    kin  = reshape(S.kL1, :, S.Ncoeff)
    kout = reshape(S.kL2, :, S.Ncoeff)
    @batch for i ∈ eachindex(view(S.λ,1,1,:))
        @views @inbounds mul!(kout[i,:], S.λ[:,:,i], kin[i,:])
    end

    @batch for i = 1:S.Ncoeff
        @views mul!(S.kL1[:,:,:,i], S.ifftplan, S.kL2[:,:,:,i])
    end

    x .= vec(@view S.kL1[idx,:])
    return x
end

function Base.:*(S::NFFTNormalOpBasisFunc, b::AbstractVector{T}) where T
    x = similar(b)
    return mul!(x, S, b)
end