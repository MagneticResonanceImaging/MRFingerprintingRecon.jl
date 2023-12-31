function scGROG(data::AbstractArray{Complex{T}}, trj) where {T}
    # self-calibrating radial GROG
    # doi: 10.1002/mrm.21565

    # data should be passed with dimensions Nr x Ns x Ncoil
    Nr = size(data, 1) #number of readout points
    Ns = size(data, 2) # number of spokes across whole trajectory
    Ncoil = size(data, 3)
    Nd = size(trj[1], 1) # number of dimensions
    @assert Nr > Ncoil "Ncoil < Nr, problem is ill posed"
    @assert Ns > Ncoil^2 "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    lnG = Array{Complex{T}}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ = Array{Complex{T}}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = dropdims(diff(trjr[:, 1:2, :], dims=2), dims=2)' .* Nr # units of sampling rate

    # 2) For each spoke, solve Eq3 for Gθ and compute matrix log
    Threads.@threads for ip ∈ axes(data, 2)
        @views Gθ = transpose(data[1:end-1, ip, :] \ data[2:end, ip, :])
        vθ[ip, :, :] = log(Gθ) # matrix log
    end

    # 3) Solve Eq8 Nc^2 times
    Threads.@threads for i ∈ CartesianIndices(@view lnG[1, :, :])
        @views lnG[:, i] .= nm \ vθ[:, i]
    end

    lnG = [lnG[id, :, :] for id = 1:Nd]

    return lnG
end

function griddedBackProjection(data::AbstractArray{Complex{T}}, lnG, trj, U::Matrix{Complex{T}}, cmaps; density=false, verbose=false) where {T}
    # performs GROG gridding, returns backprojection and kernels
    # assumes data is passed with dimensions Nr x NCyc*Nt x Ncoil

    img_shape = size(cmaps[1])
    Nr = size(data, 1) #number of readout points
    Nt = length(trj) # number of time points
    @assert Nt == size(U, 1) "Mismatch between trajectory and basis"
    Ncoeff = size(U, 2)

    idx = CartesianIndices(img_shape)
    Ncoil = length(cmaps)
    data = reshape(data, :, Nt, Ncoil) # make sure data has correct size before gridding

    exp_method = ExpMethodHigham2005()
    cache = [ExponentialUtilities.alloc_mem(lnG[1], exp_method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lnG[1]) for _ ∈ 1:Threads.nthreads()]

    # gridding
    t = @elapsed Threads.@threads for i ∈ CartesianIndices(@view data[:, :, 1])
        idt = Threads.threadid()
        for j ∈ length(img_shape):-1:1
            trj_i = trj[i[2]][j, i[1]] * img_shape[j] + 1 / 2
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[j]
            trj[i[2]][j, i[1]] = k_idx + img_shape[j] ÷ 2

            lGcache[idt] .= shift .* lnG[j]
            @views data[i, :] = exponential!(lGcache[idt], exp_method, cache[idt]) * data[i, :]
        end
    end
    verbose && println("Gridding: t = $t s"); flush(stdout)

    # backprojection & kernel calculation
    dataU = zeros(Complex{T}, img_shape..., Ncoil, Ncoeff)
    Λ = zeros(Complex{T}, Ncoeff, Ncoeff, img_shape...)
    if density
        D = zeros(Int16, img_shape..., Nt)
    end

    t = @elapsed for i ∈ CartesianIndices(@view data[:, :, 1])
        k_idx = ntuple(j -> mod1(Int(trj[i[2]][j, i[1]]) - img_shape[j]÷2, img_shape[j]), length(img_shape)) # incorporates ifftshift
        k_idx = CartesianIndex(k_idx)

        # multiply by basis for backprojection
        for icoef ∈ axes(U, 2), icoil ∈ axes(data, 3)
            @views dataU[k_idx, icoil, icoef] += data[i[1], i[2], icoil] * conj(U[i[2], icoef])
        end
        # add to kernel
        for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
            Λ[ic[1], ic[2], k_idx] += conj(U[i[2], ic[1]]) * U[i[2], ic[2]]
        end
        if density
            k_idx_D = CartesianIndex(ntuple(j -> Int(trj[i[2]][j, i[1]]), length(img_shape)))
            D[k_idx_D, i[2]] += 1
        end
    end
    verbose && println("Kernel calculation & back-projection time: t = $t s"); flush(stdout)

    # compute backprojection
    xbp = zeros(Complex{T}, img_shape..., Ncoeff)
    xbpci = [Array{Complex{T}}(undef, img_shape) for _ = 1:Threads.nthreads()]
    Threads.@threads for icoef ∈ axes(U, 2)
        idt = Threads.threadid()
        for icoil ∈ eachindex(cmaps)
            @views ifft!(dataU[idx, icoil, icoef])
            @views fftshift!(xbpci[idt], dataU[idx, icoil, icoef])
            xbp[idx, icoef] .+= conj.(cmaps[icoil]) .* xbpci[idt]
        end
    end

    if density
        return xbp, Λ, D
    else
        return xbp, Λ
    end
end