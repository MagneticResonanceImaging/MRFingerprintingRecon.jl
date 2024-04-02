function grog_calculatekernel(data, trj, Nr)
    # self-calibrating radial GROG (https://doi.org/10.1002/mrm.21565)

    Ncoil = size(data, 3)
    data = reshape(data, Nr, :, Ncoil)
    Ns = size(data, 2) # number of spokes across whole trajectory
    Nd = size(trj[1], 1) # number of dimensions

    @assert Nr > Ncoil "Ncoil < Nr, problem is ill posed"
    @assert Ns > Ncoil^2 "Number of spokes < Ncoil^2, problem is ill posed"

    # preallocations
    lnG = Array{eltype(data)}(undef, Nd, Ncoil, Ncoil) #matrix of GROG operators
    vθ  = Array{eltype(data)}(undef, Ns, Ncoil, Ncoil)

    # 1) Precompute n, m for the trajectory
    trjr = reshape(combinedimsview(trj), Nd, Nr, :)
    nm = transpose(dropdims(diff(trjr[:, 1:2, :], dims=2), dims=2)) .* Nr # units of sampling rate

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

function grog_griddata!(data, trj, Nr, img_shape)
    lnG = grog_calculatekernel(data, trj, Nr)

    Ncoil = size(data, 3)

    Nt = length(trj) # number of time points
    data = reshape(data, :, Nt, Ncoil) # make sure data has correct size before gridding

    exp_method = ExpMethodHigham2005()
    cache = [ExponentialUtilities.alloc_mem(lnG[1], exp_method) for _ ∈ 1:Threads.nthreads()]
    lGcache = [similar(lnG[1]) for _ ∈ 1:Threads.nthreads()]

    Threads.@threads for i ∈ CartesianIndices(@view data[:, :, 1])
        idt = Threads.threadid() # TODO: fix data race bug
        for j ∈ eachindex(img_shape)
            trj_i = trj[i[2]][j, i[1]] * img_shape[j] + 1 / 2 # +1/2 to avoid stepping outside of FFT definition ∈ (-img_shape[j]/2+1, img_shape[j]/2)
            k_idx = round(trj_i)
            shift = (k_idx - trj_i) * Nr / img_shape[j]

            # overwrite trj with rounded grid point index
            trj[i[2]][j, i[1]] = k_idx + img_shape[j] ÷ 2

            # overwrite data with gridded data
            lGcache[idt] .= shift .* lnG[j]
            @views data[i, :] = exponential!(lGcache[idt], exp_method, cache[idt]) * data[i, :]
        end
    end
end

function calculateBackProjection_gridded(data, trj, U, cmaps)
    Ncoil = length(cmaps)
    Ncoeff = size(U, 2)
    img_shape = size(cmaps[1])
    img_idx = CartesianIndices(img_shape)

    Nt = length(trj)
    @assert Nt == size(U, 1) "Mismatch between trajectory and basis"
    data = reshape(data, :, Nt, Ncoil)

    dataU = similar(data, img_shape..., Ncoeff)
    xbp = zeros(eltype(data), img_shape..., Ncoeff)

    Threads.@threads for icoef ∈ axes(U, 2)
        for icoil ∈ axes(data, 3)
            dataU[img_idx, icoef] .= 0

            for i ∈ CartesianIndices(@view data[:, :, 1])
                k_idx = ntuple(j -> mod1(Int(trj[i[2]][j, i[1]]) - img_shape[j] ÷ 2, img_shape[j]), length(img_shape)) # incorporates ifftshift
                k_idx = CartesianIndex(k_idx)

                @views dataU[k_idx, icoef] += data[i[1], i[2], icoil] * conj(U[i[2], icoef])
            end

            @views ifft!(dataU[img_idx, icoef])
            @views xbp[img_idx, icoef] .+= conj.(cmaps[icoil]) .* fftshift(dataU[img_idx, icoef])
        end
    end
    return xbp
end