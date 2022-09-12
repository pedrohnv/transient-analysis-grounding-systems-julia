#=
Reproducing the results in [1] for a grounding grid.

[1] L. D. Grcev and M. Heimbach, "Frequency dependent and transient
characteristics of substation grounding systems," in IEEE Transactions on
Power Delivery, vol. 12, no. 1, pp. 172-178, Jan. 1997.
doi: 10.1109/61.568238
=#
using Plots
include("../hem.jl");

function simulate(gs::Int, freq, nfrac, mhem, symmetry)
    ## Parameters
    # Soil
    mu0 = MU0;
    mur = 1.0;
    eps0 = EPS0;
    epsr = 10;
    σ1 = 1.0/1000.0;
    # Frequencies
    nf = length(freq);
    Ω = 2*pi*freq[nf];
    λ = (2*pi/Ω)*(1/sqrt( epsr*eps0*mu0/2*(1 + sqrt(1 + (σ1/(Ω*epsr*eps0))^2)) ));
    frac = λ/nfrac; #for segmentation

    # Grid
    r = 7e-3;
    h = -0.5;
    l = gs;
    n = Int(gs/10) + 1;
    num_seg = Int( ceil(gs/((n - 1)*frac)) );
    grid = Grid(n, n, l, l, num_seg, num_seg, r, h);
    electrodes, nodes = electrode_grid(grid);
    num_electrodes = ns = length(electrodes)
    num_nodes = size(nodes)[1]
    inj_node = matchrow([0.,0.,h], nodes)

    #create images
    images = Array{Electrode}(undef, ns);
    for i=1:ns
        start_point = [electrodes[i].start_point[1],
                       electrodes[i].start_point[2],
                       -electrodes[i].start_point[3]];
        end_point = [electrodes[i].end_point[1],
                     electrodes[i].end_point[2],
                         -electrodes[i].end_point[3]];
        r = electrodes[i].radius;
        images[i] = new_electrode(start_point, end_point, r);
    end

    mA, mB = incidence(electrodes, nodes);
    exci = zeros(ComplexF64, num_nodes);
    exci[inj_node] = 1.0;
    zh = Array{ComplexF64}(undef, nf);
    zl = Array{ComplexF64}(undef, (ns,ns));
    zt = Array{ComplexF64}(undef, (ns,ns));
    zli = Array{ComplexF64}(undef, (ns,ns));
    zti = Array{ComplexF64}(undef, (ns,ns));
    yn = Array{ComplexF64}(undef, (num_nodes, num_nodes));
    mC = Array{ComplexF64}(undef, (ns, num_nodes));

    # Integration Parameters
    max_eval = typemax(Int);
    req_abs_error = 1e-3;
    req_rel_error = 1e-4;
    error_norm = norm;
    if mhem
        intg_type = INTG_MHEM;
        if symmetry
            mpotzl, mpotzt = impedances_grid(grid, 0.0, 1.0, 1.0, 1.0, max_eval,
                                             req_abs_error, req_rel_error,
                                             error_norm, intg_type);
            mpotzli, mpotzti = impedances_grid(grid, 0.0, 1.0, 1.0, 1.0, max_eval,
                                                         req_abs_error, req_rel_error,
                                                         error_norm, intg_type, 1, true);
        else
            mpotzl, mpotzt = calculate_impedances(electrodes, 0.0, 1.0, 1.0, 1.0,
                                                  max_eval, req_abs_error,
                                                  req_rel_error, error_norm,
                                                  intg_type);
            mpotzli, mpotzti = impedances_images(electrodes, images,
                                                 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 max_eval, req_abs_error,
                                                 req_rel_error, error_norm,
                                                 intg_type);
        end
    else
        intg_type = INTG_DOUBLE;
    end
    ## Frequency loop
    for f = 1:nf
        #println("f = ", f)
        jw = 1.0im*TWO_PI*freq[f];
        kappa = σ1 + jw*epsr*eps0;
        k1 = sqrt(jw*mu0*kappa);
        kappa_air = jw*eps0;
        ref_t = (kappa - kappa_air)/(kappa + kappa_air);
        ref_l = ref_t;
        if mhem
            for k=1:ns
                for i=k:ns
                    rbar = norm(electrodes[i].middle_point - electrodes[k].middle_point);
                    zl[i,k] = exp(-k1*rbar)*jw*mpotzl[i,k];
                    zt[i,k] = exp(-k1*rbar)/(kappa)*mpotzt[i,k];
                    rbar = norm(electrodes[i].middle_point - images[k].middle_point);
                    zl[i,k] += ref_l*exp(-k1*rbar)*jw*mpotzli[i,k];
                    zt[i,k] += ref_t*exp(-k1*rbar)/(kappa)*mpotzti[i,k];
                    zl[k,i] = zl[i,k];
                    zt[k,i] = zt[i,k];
                end
            end
        else
            if symmetry
                impedances_grid!(zl, zt, grid, k1, jw, mur, kappa, max_eval,
                                 req_abs_error, req_rel_error, error_norm, intg_type);
                impedances_grid!(zli, zti, grid, k1, jw, mur, kappa, max_eval,
                                 req_abs_error, req_rel_error, error_norm,
                                 intg_type, 1, true);
                zl .+= ref_l.*zli;
                zt .+= ref_t.*zti;
            else
                calculate_impedances!(zl, zt, electrodes, k1, jw, mur, kappa,
                                      max_eval, req_abs_error, req_rel_error,
                                      error_norm, intg_type);
                impedances_images!(zl, zt, electrodes, images, k1, jw, mur, kappa,
                                   ref_l, ref_t, max_eval, req_abs_error,
                                   req_rel_error, error_norm, intg_type);
            end
        end
        exci .= 0.0;
        exci[inj_node] = 1.0;
        admittance!(yn, zl, zt, mA, mB, mC)
        ldiv!(lu!(yn), exci)
        zh[f] = exci[inj_node];
    end;
    return zh
end;
simulate(10, [1.0], 2, false, true); # force compilation

nf = 100;
freq = exp10.(range(2, stop=6.4, length=nf)); #logspace
nfrac = 20;
mhem = false;
symmetry = true;
#gs_arr = [10, 20, 30, 60, 120];
gs_arr = [10, 20, 30];
ng = length(gs_arr);
zh = Array{ComplexF64}(undef, nf, ng);
for i = 1:ng
    gs = gs_arr[i];
    @time zh[:,i] = simulate(gs, freq, nfrac, mhem, symmetry);
end

begin
    plot(xaxis=:log, legend=:topleft, xlabel="f (Hz)", ylabel="|Zh (Ω)|");
    for i = 1:ng
        plot!(freq, abs.(zh[:,i]), label=join(["GS ", gs_arr[i]]))
    end
    plot!()
end

begin
    plot(xaxis=:log, legend=:topleft, xlabel="f (Hz)", ylabel="Phase Zh (deg)");
    for i = 1:ng
        plot!(freq, 180/π*angle.(zh[:,i]), label=join(["GS ", gs_arr[i]]))
    end
    plot!()
end
