"""
    grog_calib(data, trj, Nr)

Perform GROG kernel calibration based on whole radial trajectory and passed data.
Calibration follows the work on self-calibrating radial GROG (https://doi.org/10.1002/mrm.21565).

# Arguments
- `data::Matrix{ComplexF32}`: Basis coefficients of subspace
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `Nr::Int`: Number of samples per read out
"""
function grog_calib(data, trj, Nr)

    Ncoil = size(data, 3)
    Nrep = size(data, 4)

    if (1 != Nrep) # Avoid error during reshape that joins rep and t dim
        data = permutedims(data, (1,2,4,3))
    end

    data = reshape(data, Nr, :, Ncoil)
    Ns = size(data, 2) # number of spokes across whole trajectory
    Nd = size(trj[1], 1) # number of dimensions

    Nr < Ncoil && @warn "Ncoil < Nr, problem is ill posed"
    Ns < Ncoil^2 && @warn "Number of spokes < Ncoil^2, problem is ill posed"
    @assert isinteger(Ns / (Nrep * length(trj))) "Mismatch between trajectory and data"

    # preallocations
    lnG = Array{eltype(data)}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ  = Array{eltype(data)}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = transpose(dropdims(diff(trjr[:, 1:2, :], dims=2), dims=2)) .* Nr # units of sampling rate
    nm = repeat(nm, outer = [Nrep]) # Stack trj in time if sampling repeats

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

"""
    grog_gridding!(data, trj, lnG, Nr, img_shape)

Perform gridding of data based on pre-calculated GROG kernel.

# Arguments
- `data::Matrix{ComplexF32}`: Basis coefficients of subspace
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `lnG::Vector{Matrix{Float32}}`: Natural logarithm of GROG kernel in all dimensions
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{Int}`: Image dimensions

# Dimensions:
- `data`:   [samples, spokes, timesteps, coils, repetitions of sampling pattern]
- `trj`:    [timesteps, repetitions][dims, samples]
- `lnG`:    [dims][Ncoils, Ncoils]

# Further:
- Ensure sampling pattern repeats in repetitions dimension!
"""
function grog_gridding!(data, trj, lnG, Nr, img_shape)

    Ncoil = size(data, 3)
    Nrep = size(data, 4)

    Nt = length(trj) # number of time points
    data = reshape(data, :, Nt, Ncoil, Nrep) # make sure data has correct size before gridding

    exp_method = ExpMethodHigham2005()
    cache = [ExponentialUtilities.alloc_mem(lnG[1], exp_method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lnG[1]) for _ ∈ 1:Threads.nthreads()]

    Threads.@threads for i ∈ CartesianIndices(@view data[:, :, 1, 1])
        idt = Threads.threadid() # TODO: fix data race bug
        for j ∈ eachindex(img_shape)
            trj_i = trj[i[2]][j, i[1]] * img_shape[j] + 1 / 2 # +1/2 to avoid stepping outside of FFT definition ∈ (-img_shape[j]/2+1, img_shape[j]/2)
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[j]

            # store rounded grid point index
            trj[i[2]][j, i[1]] = k_idx + img_shape[j] ÷ 2

            # overwrite trj with rounded grid point index.
            lGcache[idt] .= shift .* lnG[j]
            @views data[i, :, :] =  exponential!(lGcache[idt], exp_method, cache[idt]) * data[i, :, :]
        end
    end
end

"""
    radial_grog!(data, trj, Nr, img_shape)

Perform GROG kernel calibration and gridding of data in-place.

# Arguments
- `data::Matrix{ComplexF32}`: Basis coefficients of subspace
- `traj::Vector{Matrix{Float32}}`: Trajectory
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{Int}`: Image dimensions
"""
function radial_grog!(data, trj, Nr, img_shape)

    lnG = grog_calib(data, trj, Nr)

    grog_gridding!(data, trj, lnG, Nr, img_shape)
end
