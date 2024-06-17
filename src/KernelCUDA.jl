## ##########################################################################
# Various internally used CUDA kernels
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

function kernel_mul2!(S, Uc, U, ic1, ic2)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if i <= size(U, 1) && j <= size(S, 1)
            S[j,i] = Uc[i,ic1] * U[i,ic2]
    end
    return
end

function kernel_sort!(Λ, λ, λ3, kmask_indcs, ic1, ic2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    if i <= length(kmask_indcs)
        Λ[ic2,ic1,i] = λ3[kmask_indcs[i]]
        Λ[ic1,ic2,i] =  λ[kmask_indcs[i]]
    end
    return
end
