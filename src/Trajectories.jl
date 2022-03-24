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
    theta = vec(theta)
    phi   = vec(phi)

    kr = ((-Nr+1)/2:(Nr-1)/2) / Nr

    k = Matrix{T}(undef, 3, length(kr)*length(theta))
    k[1,:] = -kron(sin.(theta) .* cos.(phi), (kr .+ delay[1]))
    k[2,:] =  kron(sin.(theta) .* sin.(phi), (kr .+ delay[2]))
    k[3,:] =  kron(cos.(theta),              (kr .+ delay[3]))

    k = reshape(k, 3, :, Nt)
    kv = [k[:,:,t] for t=1:Nt]
    return k, kv
end