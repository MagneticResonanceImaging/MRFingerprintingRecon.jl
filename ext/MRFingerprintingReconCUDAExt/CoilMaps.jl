function MRFingerprintingRecon.calculateCoilMaps(data::CuArray{Complex{T}}, trj::CuArray{T}, img_shape::NTuple{N,Int}; U=CUDA.ones(T, size(trj)[end]), mask=CUDA.ones(Bool, size(trj)[2:end]), kernel_size=ntuple(_ -> 6, N), calib_size=ntuple(i -> nextprod((2, 3, 5), img_shape[i] ÷ (maximum(img_shape) ÷ 32)), length(img_shape)), eigThresh_1=0.01, eigThresh_2=0.9, nmaps=1, verbose=false) where {N,T}
    @assert all([icalib .== nextprod((2, 3, 5), icalib) for icalib ∈ calib_size]) "calib_size has to be composed of the prime factors 2, 3, and 5 (cf. NonuniformFFTs.jl documentation)."

    calib_scale = cu(collect(img_shape ./ calib_size))
    trj_idx = reshape(all(abs.(trj) .* calib_scale .< 0.5; dims=1), size(trj, 2), :)
    mask .&= trj_idx # update mask to only take calib region of k-space in CoilwiseCG
    trj  .*= calib_scale # scale trj for correct FOV

    x = MRFingerprintingRecon.calculateCoilwiseCG(data, trj, calib_size; U, mask)

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

# wrapper for 4D data arrays with readout lines in separate axis
function MRFingerprintingRecon.calculateCoilMaps(data::CuArray{Tc,4}, trj::CuArray{T,4}, img_shape::NTuple{N,Int}; mask=CUDA.ones(Bool, size(trj)[2:end]), kwargs...) where {N,T,Tc<:Complex}
    data = reshape(data, :, size(data,3), size(data,4))
    trj = reshape(trj, size(trj,1), :, size(trj,4))
    mask = reshape(mask, :, size(mask,3))
    return calculateCoilMaps(data, trj, img_shape; kwargs..., mask)
end