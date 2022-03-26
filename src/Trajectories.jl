function kooshballGA(Nr, Ncyc, Nt; delay=(0,0,0), T=Float32)
    M = [0 1 0; 0 0 1; 1 0 1]
    v = eigvecs(M)
    GA1 = real(v[1,3] / v[3,3])
    GA2 = real(v[2,3] / v[3,3])
    theta = acos.(mod.((0:(Ncyc*Nt-1))*GA1, 1))
    phi = (0:(Ncyc*Nt-1)) * 2π * GA2

    theta = reshape(theta, Nt, Ncyc)
    phi   = reshape(phi,   Nt, Ncyc)

    return kooshball(Nr, theta', phi'; delay=delay, T=T)
end

# delay is in (HF, AP, LR)
function kooshball(Nr, theta, phi; delay=(0,0,0), T=Float32)
    Ncyc, Nt = size(theta)

    kr = collect(((-Nr+1)/2:(Nr-1)/2) / Nr)
    stheta = sin.(theta)
    ctheta = cos.(theta)
    sphi   = sin.(phi)
    cphi   = cos.(phi)

    k = Vector{Matrix{T}}(undef, Nt)
    for it ∈ eachindex(k)
        ki = Array{T,3}(undef, 3, Nr, Ncyc)
        @batch for ic = 1:Ncyc, ir ∈ 1:Nr
            ki[1,ir,ic] = -stheta[ic,it] * cphi[ic,it] * (kr[ir] + delay[1])
            ki[2,ir,ic] =  stheta[ic,it] * sphi[ic,it] * (kr[ir] + delay[2])
            ki[3,ir,ic] =  ctheta[ic,it] *               (kr[ir] + delay[3])
        end
        k[it] = reshape(ki, 3, :)
    end
    return k
end