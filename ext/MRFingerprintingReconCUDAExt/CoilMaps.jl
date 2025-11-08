function MRFingerprintingRecon.calcCoilMaps(data::CuArray{Complex{T}}, trj::CuArray{T}, nsamp_t::CuArray{<:Integer}, img_shape::NTuple{N,Int}; U=CUDA.ones(Complex{T}, length(nsamp_t)), kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."
   
    # mask data and trj for coil map estimates
    calib_scale = cu(collect(img_shape ./ calib_size))
    trj_idx = vec(all(abs.(trj) .* calib_scale .< 0.5; dims=1))

    data_calib = data[trj_idx, :]
    cumsum_nsamp = trj[:, trj_idx] .* calib_scale

    # launch kernel to count masked samples per frame
    cumsum_nsamp = cu([0; cumsum(nsamp_t[1:end-1])])
    nsamp_t_calib = CUDA.zeros(eltype(nsamp_t), size(nsamp_t))

    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    threads_x = min(max_threads, maximum(nsamp_t))
    threads_y = min(max_threads ÷ threads_x, length(nsamp_t))
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (maximum(nsamp_t), length(nsamp_t)) ./ threads)
    
    @cuda threads=threads blocks=blocks count_samples!(nsamp_t_calib, nsamp_t, cumsum_nsamp, trj_idx)

    x = MRFingerprintingRecon.calculateCoilwiseCG(data_calib, cumsum_nsamp, nsamp_t_calib, calib_size; U)

    imdims = ntuple(i -> i, length(img_shape))
    kbp = fftshift(x, imdims)
    fft!(kbp, imdims)
    kbp = fftshift(kbp, imdims)

    t = @elapsed begin
        cmaps = espirit(Array(kbp), img_shape, kernel_size; eigThresh_1, eigThresh_2, nmaps)
    end
    verbose && println("espirit: $t s")

    cmaps = [cu(cmaps[CartesianIndices(img_shape), ic, in]) for ic ∈ axes(cmaps, length(img_shape) + 1), in = (nmaps == 1 ? 1 : 1:nmaps)]
    return cmaps
end

## ##########################################################################
# Internal use
#############################################################################
function count_samples!(nsamp_t_calib, nsamp_t, cumsum_nsamp, trj_idx)
    ik = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    it = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if it <= length(nsamp_t)
        ik_abs = cumsum_nsamp[it] + ik
        if ik <= nsamp_t[it] && trj_idx[ik_abs] != 0
            @inbounds CUDA.@atomic nsamp_t_calib[it] += 1
            return
        end
    end
end