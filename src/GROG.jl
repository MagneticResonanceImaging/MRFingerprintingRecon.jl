"""
    grog_calib(data, trj, Nr)

Perform GROG kernel calibration based on whole radial trajectory and passed data.
Calibration follows the work on self-calibrating radial GROG (https://doi.org/10.1002/mrm.21565).

# Arguments
- `data::AbstractAbstractArray`: Complex dataset with dimensions (samples per time frame [Nr], time frames, Rx channels)
- `trj::AbstractArray`: Trajectory with samples corresponding to the dataset, passed as AbstractArray with dimension (dims, samples per time frame, time frames)
- `Nr::Int`: Number of samples per read out
"""
function grog_calib(data, trj, Nr)
    Ncoil = size(data, 3)
    Nrep = size(data, 4)

    if (1 != Nrep)
        data = permutedims(data, (1, 2, 4, 3))
    end

    data_rs = reshape(data, Nr, :, Ncoil)
    Ns = size(data_rs, 2) # number of spokes across whole trajectory
    Nd = size(trj, 1) # number of dimensions

    Nr < Ncoil && @warn "Ncoil < Nr, problem is ill posed"
    Ns < Ncoil^2 && @warn "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    lnG = Array{eltype(data_rs)}(undef, Nd, Ncoil, Ncoil) # matrix of GROG operators
    vθ = Array{eltype(data_rs)}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trj_rs = reshape(trj, Nd, Nr, :)
    nm = transpose(dropdims(diff(trj_rs[:, 1:2, :], dims=2), dims=2)) .* Nr # units of sampling rate
    nm = repeat(nm, outer=[Nrep]) # Stack trj in time if sampling repeats

    # 2) For each spoke, solve Eq3 for Gθ and compute matrix log
    Threads.@threads for ip ∈ axes(data_rs, 2)
        @views Gθ = transpose(data_rs[1:end-1, ip, :] \ data_rs[2:end, ip, :])
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
- `data::AbstractVector{<:AbstractMatrix}`: Complex dataset passed as AbstractVector of matrices
- `trj::AbstractArray}`: Trajectory with samples corresponding to the dataset passed as AbstractVector of matrices with Float32 entries
- `lnG::AbstractArray{Matrix}`: Natural logarithm of GROG kernel in all dimensions
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{<:Integer}`: Image dimensions

# Output
- `trj::AbstractArray{<:Integer}`: Cartesian trajectory with the elements `trj[idim,ik,it] ∈ (1, img_shape[idim])`

# Dimensions:
- `data`:   (samples, timesteps, coils, repetitions of sampling pattern)
- `trj`:    (dims, samples, timesteps)
- `lnG`:    (dims][Ncoils, Ncoils)
"""
function grog_gridding!(data, trj, lnG, Nr, img_shape)
    exp_method = ExpMethodHigham2005()
    trj_cart = similar(trj, Int32) #[similar(trj_i, Int32) for trj_i ∈ trj]

    Nt = size(data, 2)
    Ncoil = size(data, 3)
    Nrep = size(data, 4)
    data = reshape(data, :, Nt, Ncoil, Nrep)

    @tasks for it ∈ axes(data, 2)
        @local begin
            exp_cache = ExponentialUtilities.alloc_mem(lnG[1], exp_method)
            lnG_cache = similar(lnG[1])
        end

        for is ∈ axes(data, 1), idim ∈ eachindex(img_shape)
            trj_i = trj[idim, is, it] * img_shape[idim] + 1 / 2 # +1/2 to avoid stepping outside of FFT definition ∈ (-img_shape[idim]/2+1, img_shape[idim]/2)
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[idim]

            # store rounded grid point index
            trj_cart[idim, is, it] = Int(k_idx) + img_shape[idim] ÷ 2

            # grid data
            lnG_cache .= shift .* lnG[idim]
            @views data[is, it, :, :] = exponential!(lnG_cache, exp_method, exp_cache) * data[is, it, :, :]
        end
    end
    return trj_cart
end

"""
    radial_grog!(data, trj, Nr, img_shape)

Perform GROG kernel calibration and gridding [1] of data in-place. The trajectory is returned with integer values.

# Arguments
- `data::AbstractArray}`: Complex dataset with dimensions (samples per time frame [Nr], time frames, Rx channels)
- `trj::AbstractArray`: Trajectory with samples corresponding to the dataset, passed as AbstractArray with dimension (dims, samples per time frame, time frames)
- `Nr::Int`: Number of samples per read out
- `img_shape::Tuple{Int}`: Image dimensions

# Output
- `trj::AbstractArray{<:Integer}}`: Cartesian trajectory with the elements `trj[idim,ik,it] ∈ (1, img_shape[idim])`

# References
1. Seiberlich N, Breuer F, Blaimer M, Jakob P, and Griswold M. "Self-calibrating GRAPPA operator gridding for radial and spiral trajectories". Magn. Reson. Med. 59 (2008), pp. 930-935. https://doi.org/10.1002/mrm.21565
"""
function radial_grog!(data, trj, Nr, img_shape)
    lnG = grog_calib(data, trj, Nr)
    trj_cart = grog_gridding!(data, trj, lnG, Nr, img_shape)
    return trj_cart
end