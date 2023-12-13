using BenchmarkTools
using MRFingerprintingRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using Test

## set parameters
T  = Float32
Nx = 64
Nc = 4
Nt = 100
Ncyc = 100
Ncoil = 4

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)

## coil maps
cmaps = zeros(Complex{T}, Nx, Nx, 4)
cmaps[:,:,1] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷4, Nx÷4),  (Nx,Nx))], 2)
cmaps[:,:,2] .= phantom(1:Nx, 1:Nx, [gauss2((Nx÷4, 3Nx÷4), (Nx,Nx))], 2)
cmaps[:,:,3] .= phantom(1:Nx, 1:Nx, [gauss2((3Nx÷4,Nx÷4),  (Nx,Nx))], 2)
cmaps[:,:,4] .= phantom(1:Nx, 1:Nx, [gauss2((3Nx÷4,3Nx÷4), (Nx,Nx))], 2)
for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data and backprojection
D = phantom(1:Nx, 1:Nx, [gauss2((Nx÷2,Nx÷2), (Nx,Nx))], 2)
D = repeat(D, 1, 1, Nt)
# D = rand(Nx, Nx, Nt) .< 0.5
# D = ones(Nx, Nx, Nt)
data = Array{Complex{T}}(undef, Nx, Nx, Nt)
xbp = similar(x) .= 0
for icoil = 1:Ncoil
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        data[i,:] .= U * x[i,:] .* cmaps[icoil][i]
    end
    data .= ifftshift(data, (1,2))
    fft!(data, (1,2))
    data .= fftshift(data, (1,2))
    data .*= D
    data .= ifftshift(data, (1,2))
    ifft!(data, (1,2))
    data .= fftshift(data, (1,2))
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        xbp[i,:] .+= U' * data[i,:] * conj(cmaps[icoil][i])
    end
end

## construct forward operator
A = FFTNormalOpBasisFuncLO((Nx,Nx), U; cmaps=cmaps, D=D)

## test forward operator is symmetric
for i ∈ CartesianIndices((Nx, Nx))
    @test A.prod!.A.Λ[:,:,i]' ≈ A.prod!.A.Λ[:,:,i] rtol = eps(T)
end

## reconstruct
xr = zeros(Complex{T}, size(vec(x)))
cg!(xr, A, vec(xbp), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## test equivalence
mask = abs.(x[:,:,1]) .> 0
@test xr[mask,:] ≈ x[mask,:] rtol = 1e-1

##
# using Plots
# plot(
#     heatmap(real.(cat(x[:,:,1], xr[:,:,1], dims=2)), aspect_ratio=1),
#     heatmap(imag.(cat(x[:,:,1], xr[:,:,1], dims=2)), aspect_ratio=1),
#     size=(800,800), layout=(2,1)
# )