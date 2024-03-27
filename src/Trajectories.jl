function kooshballGA(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), T = Float32)
    gm1, gm2 = calculateGoldenMeans()
    theta = acos.(mod.((0:(Ncyc*Nt-1)) * gm1, 1))
    phi = (0:(Ncyc*Nt-1)) * 2π * gm2

    theta = reshape(theta, Nt, Ncyc)
    phi = reshape(phi, Nt, Ncyc)

    return kooshball(Nr, theta', phi'; thetaRot = thetaRot, phiRot = thetaRot, delay = delay, T = T)
end

# delay is in (HF, AP, LR)
function kooshball(Nr, theta, phi; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), T = Float32)
    Ncyc, Nt = size(theta)

    kr = collect(((-Nr+1)/2:(Nr-1)/2) / Nr)
    stheta = sin.(theta)
    ctheta = cos.(theta)
    sphi = sin.(phi)
    cphi = cos.(phi)

    k = Vector{Matrix{T}}(undef, Nt)
    if thetaRot == 0 && phiRot == 0
        for it ∈ eachindex(k)
            ki = Array{T,3}(undef, 3, Nr, Ncyc)
            @batch for ic = 1:Ncyc, ir ∈ 1:Nr
                ki[1, ir, ic] = -stheta[ic, it] * cphi[ic, it] * (kr[ir] + delay[1])
                ki[2, ir, ic] =  stheta[ic, it] * sphi[ic, it] * (kr[ir] + delay[2])
                ki[3, ir, ic] =  ctheta[ic, it]                * (kr[ir] + delay[3])
            end
            k[it] = reshape(ki, 3, :)
            @. k[it] = max(min(k[it], 0.5), -0.5) # avoid NFFT.jl to throw errors. This should alter only very few points
        end
    else
        sthetaRot = sin(thetaRot)
        cthetaRot = cos(thetaRot)
        sphiRot   = sin(phiRot)
        cphiRot   = cos(phiRot)

        k = Vector{Matrix{T}}(undef, Nt)
        for it ∈ eachindex(k)
            ki = Array{T,3}(undef, 3, Nr, Ncyc)
            @batch for ic = 1:Ncyc, ir ∈ 1:Nr
                ki[1, ir, ic] = -(cphiRot * cphi[ic, it] * cthetaRot * stheta[ic, it] - sphiRot *  sphi[ic, it] * stheta[ic, it] + cphiRot * ctheta[ic, it] * sthetaRot)    * (kr[ir] + delay[1])
                ki[2, ir, ic] =  (cphiRot * sphi[ic, it]             * stheta[ic, it] + sphiRot * (cphi[ic, it] * cthetaRot * stheta[ic, it] + ctheta[ic, it] * sthetaRot)) * (kr[ir] + delay[2])
                ki[3, ir, ic] =  (cthetaRot * ctheta[ic, it] - sthetaRot * cphi[ic, it] * stheta[ic, it])                                                                   * (kr[ir] + delay[3])
            end
            k[it] = reshape(ki, 3, :)
            @. k[it] = max(min(k[it], 0.5), -0.5) # avoid NFFT.jl to throw errors. This should alter only very few points
        end
    end
    return k
end

function calculateGoldenMeans()

    # Calculate 3D golden means, See:
    #   Chan, R.W., Ramsay, E.A., Cunningham, C.H. and Plewes, D.B. (2009),
    #   Temporal stability of adaptive 3D radial MRI using multidimensional golden means.
    #    Magn. Reson. Med., 61: 354-363.
    #   https://doi.org/10.1002/mrm.21837
    # for more information

    M = [0 1 0; 0 0 1; 1 0 1]
    v = eigvecs(M)
    gm1 = real(v[1, 3] / v[3, 3])
    gm2 = real(v[2, 3] / v[3, 3])
    return gm1, gm2
end
