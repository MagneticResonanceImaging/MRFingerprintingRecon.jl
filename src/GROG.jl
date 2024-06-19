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

    Ncoil = size(data[1], 2)
    Nrep = size(data[1], 3)

    if (1 != Nrep) # Avoid error during reshape that joins rep and t dim
        data = cat(data..., dims = 4)
        data = permutedims(data, (1,4,3,2))
    else
        data = cat(data..., dims = 3)
        data = permutedims(data, (1,3,2))
    end

    data = reshape(data, Nr, :, Ncoil)
    data = [data[:,i,:] for i=1:size(data, 2)]

    Ns = length(data) # number of spokes across whole trajectory
    Nd = size(trj[1], 1) # number of dimensions

    Nr < Ncoil && @warn "Ncoil < Nr, problem is ill posed"
    Ns < Ncoil^2 && @warn "Number of spokes < Ncoil^2, problem is ill posed"
    @assert isinteger(Ns / (Nrep * length(trj))) "Mismatch between trajectory and data"

    # preallocations
    lnG = Array{eltype(data[1])}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ  = Array{eltype(data[1])}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = transpose(dropdims(diff(trjr[:, 1:2, :], dims=2), dims=2)) .* Nr # units of sampling rate
    nm = repeat(nm, outer = [Nrep]) # Stack trj in time if sampling repeats

    # 2) For each spoke, solve Eq3 for Gθ and compute matrix log
    Threads.@threads for ip ∈ axes(data, 1)
        @views Gθ = transpose(data[ip][1:end-1, :] \ data[ip][2:end, :])
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

    Ncoil = size(data[1], 2)
    Nrep = size(data[1], 3)

    Nt = length(trj) # number of time points

    exp_method = ExpMethodHigham2005()
    cache = [ExponentialUtilities.alloc_mem(lnG[1], exp_method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lnG[1]) for _ ∈ 1:Threads.nthreads()]

    Threads.@threads for i ∈ CartesianIndices(@view cat(data..., dims = 4)[:, 1, 1, :])
        idt = Threads.threadid() # TODO: fix data race bug
        for j ∈ eachindex(img_shape)
            trj_i = trj[i[2]][j, i[1]] * img_shape[j] + 1 / 2 # +1/2 to avoid stepping outside of FFT definition ∈ (-img_shape[j]/2+1, img_shape[j]/2)
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[j]

            # store rounded grid point index
            trj[i[2]][j, i[1]] = k_idx + img_shape[j] ÷ 2

            # overwrite trj with rounded grid point index.
            lGcache[idt] .= shift .* lnG[j]
            @views data[i[2]][i[1], :, :] =  exponential!(lGcache[idt], exp_method, cache[idt]) * data[i[2]][i[1], :, :]
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
