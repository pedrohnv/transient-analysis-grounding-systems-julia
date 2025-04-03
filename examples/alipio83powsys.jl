#=
Reproducing some results for the electric field on ground level for an
horizontal electrode with frequency independent of the soil parameters.

[1] R.S. Alipio, M.A.O. Schroeder, M.M. Afonso, T.A.S. Oliveira, S.C. Assis,
Electric fields of grounding electrodes with frequency dependent soil parameters,
Electric Power Systems Research,
Volume 83, Issue 1, 2012, Pages 220-226, ISSN 0378-7796,
https://doi.org/10.1016/j.epsr.2011.11.011.
(http://www.sciencedirect.com/science/article/pii/S0378779611002781)
=#
using FLoops
using Plots

include("../hem.jl");


function run_case(lmax, sigma0)
    mur = 1.0  # soil rel. magnetic permeability
    # Integration Parameters
    max_eval = typemax(Int)
    req_abs_error = 1e-5
    req_rel_error = 1e-6
    error_norm = norm
    intg_type = INTG_MHEM

    # electrodes
    r = 7e-3
    L = 15.0
    h = 1.0
    x0 = 2.5
    ne = ceil(Int, L / lmax) + 1
    nn = ne + 1;
    println("Num. segments = ", ne)
    println("Num. nodes = ", nn)
    elec = new_electrode([x0, 0., -h], [x0 + L, 0., -h], r)
    electrodes, nodes = segment_electrode(elec, ne)
    images = deepcopy(electrodes)
    for i in eachindex(images)
        images[i].start_point[3] *= -1
        images[i].end_point[3] *= -1
        images[i].middle_point[3] *= -1
    end
    inj_node = 1

    # frequencies
    freq = [100.0, 500e3, 1e6, 2e6]
    ns = length(freq)
    s = TWO_PI * freq * 1.0im
    
   # define an array of points where to calculate quantities
   # line along X axis from 0 to 20
   dr = 0.5
   num_points = ceil(Int, 20.0 / dr) + 1
   println("Num. points to calculate GPD = ", num_points)
   points = zeros(3, num_points)
   for i = 1:num_points
       points[1, i] = dr * i
   end

    # malloc matrices =========================================================
    gpr_s = zeros(ComplexF64, ns)
    np_field = num_points
    efield_s = zeros(ComplexF64, 3, np_field, ns)
    # efield[i,j,k] => i-field, j-point, k-frequency
    # Incidence and "z-potential" (mHEM) matrices =============================
    mA, mB = incidence(electrodes, nodes)
    # calculate distances to avoid repetition
    rbar = Array{Float64}(undef, (ne,ne))
    rbari = copy(rbar)
    for k = 1:ne
        p1 = collect(electrodes[k].middle_point)
        for i = k:ne
            p2 = collect(electrodes[i].middle_point)
            p3 = collect(images[i].middle_point)
            rbar[i,k] = norm(p1 - p2)
            rbari[i,k] = norm(p1 - p3)
        end
    end
    mpotzl, mpotzt = calculate_impedances(electrodes, 0.0, 1.0, 1.0, 1.0;
                                          max_eval, req_abs_error,
                                          req_rel_error, error_norm, intg_type)
    mpotzli, mpotzti = impedances_images(electrodes, images,
                                         0.0, 1.0, 1.0, 1.0, 1.0, 1.0;
                                         max_eval, req_abs_error,
                                         req_rel_error, error_norm, intg_type)
    @floop for f = 1:ns
        @init begin
            zl = similar(mpotzl)
            zt = similar(mpotzl)
            yn = Array{ComplexF64}(undef, nn, nn)
            yla = Array{ComplexF64}(undef, ne, nn)
            ytb = similar(yla)
            ie = Array{ComplexF64}(undef, nn)
            il = Array{ComplexF64}(undef, ne)
            it = similar(il)
        end
        # soil parameters
        sigma = sigma0
        epsr = 4.0
        kappa = (sigma + s[f] * epsr * EPS0)  # soil complex conductivity
        gamma = sqrt(s[f] * MU0 * kappa)
        iwu_4pi = s[f] * mur * MU0 / (FOUR_PI)
        one_4pik = 1.0 / (FOUR_PI * kappa)
        # reflection coefficient, soil to air
        ref_t = (kappa - s[f] * EPS0) / (kappa + s[f] * EPS0)
        ref_l = 1.0
        # modified HEM (mHEM):
        for k=1:ne
            for i=k:ne
                zl[i,k] = exp(-gamma * rbar[i,k]) * iwu_4pi * mpotzl[i,k]
                zt[i,k] = exp(-gamma * rbar[i,k]) * one_4pik * mpotzt[i,k]
                zl[i,k] += ref_l * exp(-gamma * rbari[i,k]) * iwu_4pi * mpotzli[i,k]
                zt[i,k] += ref_t * exp(-gamma * rbari[i,k]) * one_4pik * mpotzti[i,k]
            end
        end

        # Invert ZL and ZT
        zl, ipiv, info = LAPACK.sytrf!('L', zl);
        LAPACK.sytri!('L', zl, ipiv);
        zt, ipiv, info = LAPACK.sytrf!('L', zt);
        LAPACK.sytri!('L', zt, ipiv);
        # Calculate YN
        BLAS.symm!('L', 'L', complex(1.0), zl, mA, complex(0.0), yla)  # yla := inv(zl)*mA + yla*0
        BLAS.gemm!('T', 'N', complex(1.0), mA, yla, complex(0.0), yn)  # yn := mAT*yla + yn*0
        BLAS.symm!('L', 'L', complex(1.0), zt, mB, complex(0.0), ytb)  # ytb := inv(zt)*mB + ytb*0
        BLAS.gemm!('T', 'N', complex(1.0), mB, ytb, complex(1.0), yn)  # yn := mBT*ytb + yn
        ie .= 0.0
        ie[inj_node] = 1.0
        ldiv!(lu!(yn), ie)  # Solve YN * UN = IE
        gpr_s[f] = ie[inj_node]
        BLAS.gemv!('N', complex(1.0), yla, ie, complex(0.0), il)  # IL = yla * UN
        BLAS.gemv!('N', complex(1.0), ytb, ie, complex(0.0), it)  # IT = ytb * UN
        # images' effect as (1 + ref) because we're calculating on ground level
        il .*= (1.0 + ref_l)
        it .*= (1.0 + ref_t)
        # Calculate electric field

        for k = 1:num_points
            efield_s[:,k,f] .= electric_field(points[:,k], electrodes, il, it, gamma, s[f], 1.0, kappa)
        end
    end
    return gpr_s, efield_s
end
precompile(run_case, (Float64, Float64))


lmax=0.5
sigma0=1000.0
@time gpr_s, efield_s = run_case(lmax, sigma0)

p = plot()
for f = 1:size(efield_s)[3]
    plot!(abs.(efield_s[1,:,f]))
end
p