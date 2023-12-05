function scGROG(data::AbstractArray{Complex{T}}, trj) where {T}
    # self-calibrating radial GROG
    # doi: 10.1002/mrm.21565

    # data should be passed with dimensions Nr x Ns x Ncoil
    Nr = size(data,1) #number of readout points
    Ns = size(data, 2) # number of spokes across whole trajectory
    Ncoil = size(data, 3)
    Nd = size(trj[1],1) # number of dimensions
    Δk = 2/Nr
    @assert Nr > Ncoil "Ncoil < Nr, problem is ill posed"
    @assert Ns > Ncoil^2 "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    G = Array{Complex{T}}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ = Array{Complex{T}}(undef, Ns, Ncoil, Ncoil)
    Gθ = [Array{Complex{T}}(undef, Ncoil, Ncoil) for _ = 1:Threads.nthreads()]

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = dropdims(diff(trjr[:,1:2,:],dims=2),dims=2)' ./ Δk #nyquist units

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

    return G
end

# function GROGgrid(data::AbstractArray{T}, trj, U, cmaps; verbose = false) where {T}

#     Λ = Array{Complex{T}}(undef, Ncoeff, Ncoeff, img_shape...)
#     return xbp, Λ
# end