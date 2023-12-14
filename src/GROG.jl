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

function griddedBackProjection(data::AbstractArray{Complex{T}}, G, trj, U::Matrix{Complex{T}}, cmaps=(1,); density = false, verbose = false) where {T}
    # performs GROG gridding, returns backprojection and kernels
    # assumes data is passed with dimensions Nr x NCyc*Nt x Ncoil

    Nr = size(data, 1) # readout length
    Nt = length(trj) # number of time points
    @assert Nt == size(U, 1) "Mismatch between trajectory and basis"
    Ncoeff = size(U, 2)
    Nd = size(trj[1],1) # number of dimensions
    @assert Nd == size(trj[1], 1) "Mismatch between trajectory and image shape"
    # img_shape = size(cmaps[1])
    img_shape = ntuple(_ -> Nr÷2, Nd)
    idx = CartesianIndices(img_shape)
    Ncoil = length(cmaps)
    grid = T.(range(-0.5,0.5, Nr÷2)) # cartesian k-space grid
    data = reshape(data, :, Nt, Ncoil) # make sure data has correct size before gridding

    # preallocations
    ig = [Array{Int16}(undef, Nd) for _ = 1:Threads.nthreads()]
    shift = Array{T}(undef, Threads.nthreads())
    Gshift = [Array{Complex{T}}(undef, Ncoil, Ncoil) for _ = 1:Threads.nthreads()]
    data_temp = [Array{Complex{T}}(undef, Ncoil) for _ = 1:Threads.nthreads()]
    dataU = zeros(Complex{T}, img_shape..., Ncoil, Ncoeff)
    Λ = zeros(Complex{T}, Ncoeff, Ncoeff, img_shape...)
    if density
        D = zeros(Int16, img_shape..., Nt)
    end

    # gridding backprojection & kernel calculation
    t = @elapsed begin
        Threads.@threads for it ∈ axes(data,2) # iterate over time dimension
            idt = Threads.threadid()
            for ir ∈ axes(data,1) # iterate over the whole trajectory
                data_temp[idt] .= data[ir,it,:] # copy the coil data
                for j = Nd:-1:1 # apply GROG kernels in reverse order for consistency with calibration
                    _, ig[idt][j] = findmin(abs.(trj[it][j,ir] .- grid))
                    shift[idt] = (grid[ig[idt][j]] - trj[it][j,ir]) * Nr #nyquist units
                    Gshift[idt] .= G[j]^shift[idt] # this seems to be more stable than combining with next line
                    data_temp[idt] .= Gshift[idt] * data_temp[idt]
                end
                # multiply by basis for backprojection
                for icoef ∈ axes(U,2)
                    dataU[ig[idt]...,:,icoef] .+= data_temp[idt] .* conj(U[it,icoef])
                end
                # add to kernel
                for ic ∈ CartesianIndices((Ncoeff, Ncoeff))
                    Λ[ic[1],ic[2],ig[idt]...] += conj.(U[it,ic[1]]) * U[it,ic[2]]
                end
                if density
                    D[ig[idt]...,it] += 1
                end
            end
        end
    end
    verbose && println("Gridding time: t = $t s"); flush(stdout)
    Λ .= ifftshift(Λ, 3:(3+length(img_shape)-1))

    # compute backprojection
    xbp = zeros(Complex{T}, img_shape..., Ncoeff)
    xbpci = [Array{Complex{T}}(undef, img_shape...) for _ = 1:Threads.nthreads()]
    Threads.@threads for icoef ∈ axes(U,2)
        idt = Threads.threadid()
        for icoil ∈ eachindex(cmaps)
            xbpci[idt] .= ifftshift(dataU[idx,icoil,icoef])
            ifft!(xbpci[idt])
            xbpci[idt] .= fftshift(xbpci[idt])
            xbp[idx,icoef] .+= conj.(cmaps[icoil]) .* xbpci[idt]
        end
    end

    if density
        return xbp, Λ, D
    else
        return xbp, Λ
    end
end