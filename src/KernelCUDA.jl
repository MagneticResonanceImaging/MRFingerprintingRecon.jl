## ##########################################################################
# Internally used CUDA kernels
#############################################################################
function kernel_mul!(kL2_rs, Λ, kL1_rs, kmask_indcs)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= length(kmask_indcs) && j <= size(kL2_rs, 2)

        ind = kmask_indcs[i]
        tmp = 0

        for k in 1:size(Λ, 2)
            tmp += Λ[j, k, i] * kL1_rs[ind, k]
        end
        kL2_rs[ind, j] = tmp
    end
    return
end
