function scGROG(data::AbstractArray{Complex{T}}, trj) where {T}
    # self-calibrating radial GROG
    # doi: 10.1002/mrm.21565

    # data should be passed with dimensions Nr x Ns x Ncoil
    Nr = size(data,1) #number of readout points
    Ns = size(data, 2) # number of spokes across whole trajectory
    Ncoil = size(data, 3)
    Nd = size(trj[1],1) # number of dimensions
    @assert Nr > Ncoil "Ncoil < Nr, problem is ill posed"
    @assert Ns > Ncoil^2 "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    G = Array{Complex{T}}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ = Array{Complex{T}}(undef, Ns, Ncoil, Ncoil)
    Gθ = [Array{Complex{T}}(undef, Ncoil, Ncoil) for _ = 1:Threads.nthreads()]

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = dropdims(diff(trjr[:,1:2,:],dims=2),dims=2)' .* Nr #nyquist units

    # 2) For each spoke, solve Eq3 for Gθ and compute matrix log
    Threads.@threads for ip ∈ axes(data,2)
        @views Gθ[Threads.threadid()] .= transpose(data[1:end-1,ip,:] \ data[2:end,ip,:])
        @views vθ[ip,:,:] .= log(Gθ[Threads.threadid()]) # matrix log
    end

    # 3) Solve Eq8 Nc^2 times
    Threads.@threads for i ∈ CartesianIndices(@view G[1,:,:])
        @views G[:,i] .= nm \ vθ[:,i]
    end

    # 4) Use Eq9 to form Gx, Gy, Gz
    Threads.@threads for id ∈ axes(G,1)
        @views G[id,:,:] .= exp(G[id,:,:]) #matrix exponent
    end

    G = [G[id,:,:] for id=1:Nd]

    return G
end

function griddedBackProjection(data::AbstractArray{Complex{T}}, G, trj, U::Matrix{Complex{T}}, cmaps=(1,); shape = ntuple(_ -> size(data, 1)÷2, size(trj[1],1)), density = false, verbose = false) where {T}
    # performs GROG gridding, returns backprojection and kernels
    # assumes data is passed with dimensions Nr x NCyc*Nt x Ncoil
    lG = [log(Gi) for Gi ∈ G]

    Nt = length(trj) # number of time points
    @assert Nt == size(U, 1) "Mismatch between trajectory and basis"
    Ncoeff = size(U, 2)

    idx = CartesianIndices(shape)
    Ncoil = length(cmaps)
    data = reshape(data, :, Nt, Ncoil) # make sure data has correct size before gridding

    # preallocations
    dataU = zeros(Complex{T}, shape..., Ncoil, Ncoeff)
    Λ = zeros(Complex{T}, Ncoeff, Ncoeff, shape...)
    if density
        D = zeros(Int16, shape..., Nt)
    end

    method = ExpMethodHigham2005()
    cache   = [ExponentialUtilities.alloc_mem(lG[1], method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lG[1]) for _ ∈ 1:Threads.nthreads()]

    # gridding backprojection & kernel calculation
    t = @elapsed begin
        Threads.@threads for i ∈ CartesianIndices(@view data[:,:,1])
            idt = Threads.threadid()
            for j ∈ eachindex(shape)
                trj_i = trj[i[2]][j,i[1]] * shape[j] + 1/2
                ig = round(trj_i)
                shift = ig - trj_i
                trj[i[2]][j,i[1]] = ig + shape[j] ÷ 2

                lGcache[idt] .= shift .* lG[j]
                @views data[i,:] = exponential!(lGcache[idt], method, cache[idt]) * data[i,:]
            end
        end

        for i ∈ CartesianIndices(@view data[:,:,1])
            ig = CartesianIndex(ntuple(j -> Int(trj[i[2]][j,i[1]]), length(shape)))

            # multiply by basis for backprojection
            for icoef ∈ axes(U,2)
                @views dataU[ig,:,icoef] .+= data[i[1],i[2],:] .* conj(U[i[2],icoef])
            end
            # add to kernel
            for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
                Λ[ic[1],ic[2],ig] += conj.(U[i[2],ic[1]]) * U[i[2],ic[2]]
            end
            if density
                D[ig,i[2]] += 1
            end
        end
    end
    verbose && println("Gridding time: t = $t s"); flush(stdout)
    Λ .= ifftshift(Λ, 3:(3+length(shape)-1))

    # compute backprojection
    xbp = zeros(Complex{T}, shape..., Ncoeff)
    xbpci = [Array{Complex{T}}(undef, shape...) for _ = 1:Threads.nthreads()]
    Threads.@threads for icoef ∈ axes(U,2)
        idt = Threads.threadid()
        for icoil ∈ eachindex(cmaps)
            @views ifftshift!(xbpci[idt], dataU[idx,icoil,icoef])
            ifft!(xbpci[idt])
            xbpci[idt] = fftshift(xbpci[idt])
            xbp[idx,icoef] .+= conj.(cmaps[icoil]) .* xbpci[idt]
        end
    end

    if density
        return xbp, Λ, D
    else
        return xbp, Λ
    end
end