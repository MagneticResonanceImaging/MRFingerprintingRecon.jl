"""
    grog_calib(data, trj, Nr)

Perform GROG kernel calibration based on whole radial trajectory and passed data.
Calibration follows the work on self-calibrating radial GROG (https://doi.org/10.1002/mrm.21565).

# Arguments
- `data::AbstractVector{<:AbstractMatrix{cT}}`: Complex dataset passed as AbstractVector of matrices
- `trj::Vector{Matrix{Float32}}`: Trajectory with samples corresponding to the dataset passed as AbstractVector of matrices with Float32 entries
- `Nr::Int`: Number of samples per read out
"""
function grog_calib(data, trj, Nr)
    Ncoil = size(data[1], 2)
    Nrep  = size(data[1], 3)

    data_r = [reshape(data_i, Nr, :, Ncoil, Nrep) for data_i ∈ data]
    data_r = @views [vec([data_i[:,is,:,irep] for is ∈ axes(data_i, 2), irep ∈ axes(data_i, 4)]) for data_i ∈ data_r]
    data_r = reduce(vcat, data_r)

    Ns = length(data_r) # number of spokes across whole trajectory
    Nd = size(trj[1], 1) # number of dimensions

    Nr < Ncoil && @warn "Ncoil < Nr, problem is ill posed"
    Ns < Ncoil^2 && @warn "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    lnG = Array{eltype(data_r[1])}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ  = Array{eltype(data_r[1])}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trj_r = [reshape(trj_i, size(trj_i,1), Nr, :) for trj_i ∈ trj]
    nm = @views [vec([(trj_i[:,2,is] .- trj_i[:,1,is]) .* Nr for is ∈ axes(trj_i,3), _ ∈ 1:Nrep]) for trj_i ∈ trj_r]
    nm = reduce(vcat, nm)
    nm = [nm[ip][id] for ip ∈ eachindex(nm), id ∈ eachindex(nm[1])]

    # 2) For each spoke, solve Eq3 for Gθ and compute matrix log
    Threads.@threads for ip ∈ eachindex(data_r)
        @views Gθ = transpose(data_r[ip][1:end-1, :] \ data_r[ip][2:end, :])
        vθ[ip, :, :] = log(Gθ) # matrix log
    end

    # 3) Solve Eq8 Nc^2 times
    Threads.@threads for i ∈ CartesianIndices(@view lnG[1, :, :])
        @views lnG[:, i] .= nm \ vθ[:, i]
    end

    lnG = [lnG[id, :, :] for id = 1:Nd]
    return lnG
end

"""
    grog_gridding!(data, trj, lnG, Nr, img_shape)

Perform gridding of data based on pre-calculated GROG kernel.

# Arguments
- `data::AbstractVector{<:AbstractMatrix{cT}}`: Complex dataset passed as AbstractVector of matrices
- `trj::Vector{Matrix{Float32}}`: Trajectory with samples corresponding to the dataset passed as AbstractVector of matrices with Float32 entries
- `lnG::Vector{Matrix{Float32}}`: Natural logarithm of GROG kernel in all dimensions
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{Int}`: Image dimensions

- `trj::Vector{Matrix{Int32}}`: Trajectory

# Dimensions:
- `data`:   [timesteps][samples, spokes, coils, repetitions of sampling pattern]
- `trj`:    [timesteps][dims, samples, repetitions]
- `lnG`:    [dims][Ncoils, Ncoils]
"""
function grog_gridding!(data, trj, lnG, Nr, img_shape)

    Ncoil = size(data[1], 2)
    Nrep  = size(data[1], 3)

    Nt = length(trj) # number of time points

    exp_method = ExpMethodHigham2005()
    cache = [ExponentialUtilities.alloc_mem(lnG[1], exp_method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lnG[1]) for _ ∈ 1:Threads.nthreads()]

    trj_cart = [similar(trj_i, Int32) for trj_i ∈ trj]
    Threads.@threads for it ∈ eachindex(data, trj)
        idt = Threads.threadid() # TODO: fix data race bug
        for is ∈ axes(data[it],1), idim ∈ eachindex(img_shape)
            trj_i = trj[it][idim, is] * img_shape[idim] + 1 / 2 # +1/2 to avoid stepping outside of FFT definition ∈ (-img_shape[idim]/2+1, img_shape[idim]/2)
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[idim]

            # store rounded grid point index
            trj_cart[it][idim, is] = k_idx + img_shape[idim] ÷ 2

            # overwrite trj with rounded grid point index.
            lGcache[idt] .= shift .* lnG[idim]
            @views data[it][is, :, :] = exponential!(lGcache[idt], exp_method, cache[idt]) * data[it][is, :, :]
        end
    end
    return trj_cart
end

"""
    radial_grog!(data, trj, Nr, img_shape)

Perform GROG kernel calibration and gridding [1] of data in-place. The trajectory is returned with integer values.

# Arguments
- `data::AbstractVector{<:AbstractMatrix{cT}}`: Complex dataset passed as AbstractVector of matrices
- `trj::Vector{Matrix{Float32}}`: Trajectory with samples corresponding to the dataset passed as AbstractVector of matrices with Float32 entries
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{Int}`: Image dimensions

# return
- `trj::Vector{Matrix{Int32}}`: Trajectory

# References
[1] Seiberlich, N., Breuer, F., Blaimer, M., Jakob, P. and Griswold, M. (2008), Self-calibrating GRAPPA operator gridding for radial and spiral trajectories. Magn. Reson. Med., 59: 930-935. https://doi.org/10.1002/mrm.21565
"""
function radial_grog!(data, trj, Nr, img_shape)
    lnG = grog_calib(data, trj, Nr)
    trj_cart = grog_gridding!(data, trj, lnG, Nr, img_shape)
    return trj_cart
end
