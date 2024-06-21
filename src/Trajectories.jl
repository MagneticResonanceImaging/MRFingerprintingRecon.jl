"""
    traj_2d_cartesian(Nx, Ny, Nz, Nt; samplingRate_units = true, T = Float32)

Function to calculate a 2D cartesian trajectory in units of sampling rate ∈ {x | -N/2+1 ≤ x ≤ N/2 and x ∈ Z}.
With `samplingRate_units = false` the ouput is relative with samples ∈ [-1/2:1/2].

# Arguments
- `Nx::Int`: Number of frequency encoded samples per read out
- `Ny::Int`: Number of phase encoding lines
- `Nz::Int`: Number of phase encoding lines (third dimension)
- `Nt::Int`: Number of times the sampling pattern is repeated
- `samplingRate_units::Boolean`: Parameter setting the output units to sampling rate
- `T::Type`: Type defining the output units of the trajectory
"""
function traj_2d_cartesian(Nx, Ny, Nz, Nt; samplingRate_units = true, T = Float32)

    kx = collect(((-Nx+1)/2:(Nx-1)/2) / (samplingRate_units ? 1 : Nx))
    ky = collect(((-Ny+1)/2:(Ny-1)/2) / (samplingRate_units ? 1 : Ny))
    kz = collect(((-Nz+1)/2:(Nz-1)/2) / (samplingRate_units ? 1 : Nz))

    k = Vector{Matrix{T}}(undef, Nt)

    for it ∈ eachindex(k)

        ki = Array{T,4}(undef, 3, Nx, Ny, Nz)

        @batch for x ∈ 1:Nx, y ∈ 1:Ny, z ∈ 1:Nz

            ki[1, x, y, z] = kx[x]
            ki[2, x, y, z] = ky[y]
            ki[3, x, y, z] = kz[z]
        end

        k[it] = reshape(ki, 3, :)

        if samplingRate_units
            k[it] = round.(k[it] .+ 0.5) # ceil operation on arrays
        end
    end

    return k
end

"""
    kooshballGA(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), T = Float32)

Function to calculate  golden means [1] based kooshball trajectory.

# Arguments
- `Nr::Int`: Number of read out samples
- `Ncyc::Int`: Number of cycles
- `Nt::Int`: Number of time steps in the trajectory
- `thetaRot::Float`: Fixed rotation angle along theta
- `phiRot::Float`: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}`: Gradient delays in (HF, AP, LR)
- `T::Type`: Type defining the output units of the trajectory

# References
[1] Chan, R.W., Ramsay, E.A., Cunningham, C.H. and Plewes, D.B. (2009), Temporal stability of adaptive 3D radial MRI using multidimensional golden means. Magn. Reson. Med., 61: 354-363. https://doi.org/10.1002/mrm.21837
"""
function kooshballGA(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0))
    gm1, gm2 = calculateGoldenMeans()
    theta = acos.(mod.((0:(Ncyc*Nt-1)) * gm1, 1))
    phi = (0:(Ncyc*Nt-1)) * 2π * gm2

    theta = reshape(theta, Nt, Ncyc)
    phi = reshape(phi, Nt, Ncyc)

    return kooshball(Nr, theta', phi'; thetaRot = thetaRot, phiRot = phiRot, delay = delay)
end

"""
    traj_2d_radial_goldenratio(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), N = 1, T = Float32)

Function to calculate 2D golden ratio based trajectory [1].
By modifying `N` also tiny golden angles [2] are supported.

# Arguments
- `Nr::Int`: Number of read out samples
- `Ncyc::Int`: Number of cycles
- `Nt::Int`: Number of time steps in the trajectory
- `thetaRot::Float`: Fixed rotation angle along theta
- `phiRot::Float`: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}`: Gradient delays in (HF, AP, LR)
- `N::Int`: Number of tiny golden angle
- `T::Type`: Type defining the output units of the trajectory

# References
[1] Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal radial profile order based on the Golden Ratio for time-resolved MRI. IEEE TMI 26:68--76 (2007)
[2] Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V. Golden ratio sparse MRI using tiny golden angles. Magn Reson Med 75:2372-2378 (2016)
"""
function traj_2d_radial_goldenratio(Nr, Ncyc, Nt; thetaRot = 0, phiRot = 0, delay = (0, 0, 0), N = 1)

    τ = (sqrt(5) + 1) / 2

    phi = (0:(Ncyc*Nt-1)) * π / (τ + N - 1)
    phi = reshape(phi, Nt, Ncyc)

    theta = similar(phi)
    theta .= π/2 # 2D

    trj = kooshball(Nr, theta', phi'; thetaRot = thetaRot, phiRot = phiRot, delay = delay)
    trj = [trj[i][1:2,:] for i ∈ eachindex(trj)] # remove 3rd dimenion
    return trj
end

"""
    kooshball(Nr, theta, phi; thetaRot = 0, phiRot = 0, delay = (0, 0, 0))

Function to calculate kooshball trajectory.

# Arguments
- `Nr::Int`: Number of read out samples
- `theta::Array{Float,2}`: Array with dimensions: `Ncyc, Nt` defining the angles `theta` for each cycle and timestep.
- `phi::Array{Float,2}`: Array with dimensions: `Ncyc, Nt` defining the angles `phi` for each cycle and timestep.
- `thetaRot::Float`: Fixed rotation angle along theta
- `phiRot::Float`: Fixed rotation angle along phi
- `delay::Tuple{Float, Float, Float}`: Gradient delays in (HF, AP, LR)
"""
function kooshball(Nr, theta, phi; thetaRot = 0, phiRot = 0, delay = (0, 0, 0))

    @assert (eltype(theta) == eltype(phi)) "Mismatch between input types of `theta` and `phi`"

    Ncyc, Nt = size(theta)

    kr = collect(((-Nr+1)/2:(Nr-1)/2) / Nr)
    stheta = sin.(theta)
    ctheta = cos.(theta)
    sphi = sin.(phi)
    cphi = cos.(phi)

    k = Vector{Matrix{eltype(theta)}}(undef, Nt)
    if thetaRot == 0 && phiRot == 0
        for it ∈ eachindex(k)
            ki = Array{eltype(theta),3}(undef, 3, Nr, Ncyc)
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

        k = Vector{Matrix{eltype(theta)}}(undef, Nt)
        for it ∈ eachindex(k)
            ki = Array{eltype(theta),3}(undef, 3, Nr, Ncyc)
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

## ############################################
# Helper Functions
###############################################

"""
    calculateGoldenMeans()

Function to calculate 3D golden means [1].

# References
[1] Chan, R.W., Ramsay, E.A., Cunningham, C.H. and Plewes, D.B. (2009), Temporal stability of adaptive 3D radial MRI using multidimensional golden means. Magn. Reson. Med., 61: 354-363. https://doi.org/10.1002/mrm.21837
"""
function calculateGoldenMeans()

    M = [0 1 0; 0 0 1; 1 0 1]
    v = eigvecs(M)
    gm1 = real(v[1, 3] / v[3, 3])
    gm2 = real(v[2, 3] / v[3, 3])
    return gm1, gm2
end
