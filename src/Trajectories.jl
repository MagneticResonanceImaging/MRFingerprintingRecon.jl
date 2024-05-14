function traj_2d_cartesian(Nx, Ny, Nz, Nt; T = Float32)

    # Cartesian Sampling Trajectory
    #
    # Set Nx, Ny, or Nz to 1 if dimension should not be sampled

    # Define sampling on each axis
    kx = collect(((-Nx+1)/2:(Nx-1)/2) / Nx)
    ky = collect(((-Ny+1)/2:(Ny-1)/2) / Ny)
    kz = collect(((-Nz+1)/2:(Nz-1)/2) / Nz)

    k = Vector{Matrix{T}}(undef, Nt)

    for it ∈ eachindex(k)

        ki = Array{T,4}(undef, 3, Nx, Ny, Nz)

        @batch for x ∈ 1:Nx, y ∈ 1:Ny, z ∈ 1:Nz

            ki[1, x, y, z] = kx[x]
            ki[2, x, y, z] = ky[y]
            ki[3, x, y, z] = kz[z]
        end

        k[it] = reshape(ki, 3, :)
    end

    return k
end

function cartesian_in_usr(Nx, Ny, Nz, Nt; T = Float32)

    # Cartesian Sampling Trajectory
    #
    # Output traj in units of sampling rate

    trj = cartesian(Nx, Ny, Nz, Nt; T = T)

    for it ∈ eachindex(trj)

        @views mul!(trj[it][1,:], trj[it][1,:], Nx)
        @views mul!(trj[it][2,:], trj[it][2,:], Ny)
        @views mul!(trj[it][3,:], trj[it][3,:], Nz)

        trj[it] = ceil.(Int, trj[it])
    end

    return trj
end

function kooshballGA(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), T = Float32)
    gm1, gm2 = calculateGoldenMeans()
    theta = acos.(mod.((0:(Ncyc*Nt-1)) * gm1, 1))
    phi = (0:(Ncyc*Nt-1)) * 2π * gm2

    theta = reshape(theta, Nt, Ncyc)
    phi = reshape(phi, Nt, Ncyc)

    return kooshball(Nr, theta', phi'; thetaRot = thetaRot, phiRot = phiRot, delay = delay, T = T)
end


function traj_2d_radial_goldenratio(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), N = 1, T = Float32)

    # Golden ratio based 2D trajectory
    #
    # Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
    # An optimal radial profile order based on the Golden Ratio
    # for time-resolved MRI. IEEE TMI 26:68--76 (2007)

    # Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V.
    # Golden ratio sparse MRI using tiny golden angles.
    # Magn Reson Med 75:2372-2378 (2016)
    #
    #   N := number of tiny angle

    τ = (sqrt(5) + 1) / 2

    theta = 0 * (0:(Ncyc*Nt-1)) .+ π/2 # 2D only
    theta = reshape(theta, Nt, Ncyc)

    phi = (0:(Ncyc*Nt-1)) * π / (τ + N - 1) # FIXME: Float32 results in minor differences for very long repetition trains
    phi = reshape(phi, Nt, Ncyc)

    return kooshball(Nr, theta', phi'; thetaRot = thetaRot, phiRot = phiRot, delay = delay, T = T)
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
