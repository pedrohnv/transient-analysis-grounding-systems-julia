#=
Pure Julia implementation of the Hybrid Electromagnetic Model.
TODO exploit Column Major ordering for better performance
=#
using LinearAlgebra;
using HCubature;

TWO_PI = 6.283185307179586;
FOUR_PI = 12.56637061435917;
MU0 = 1.256637061435917e-6; #permeability vac.
EPS0 = 8.854187817620e-12; #permittivity vac.

struct Electrode
    start_point::Array{Float64,1}
    end_point::Array{Float64,1}
    middle_point::Array{Float64,1}
    length::Float64
    radius::Float64
    zi::Complex{Float64}
end;

function new_electrode(start_point, end_point, radius, internal_impedance)
    return Electrode(start_point, end_point, (start_point + end_point)/2.0,
                     norm(start_point - end_point), radius, internal_impedance)
end;

function segment_electrode(electrode::Electrode, num_segments::Int)
    nn = num_segments + 1;
    nodes = Array{Float64}(undef, nn, 3)
    startp = Array{Float64,1}(undef, 3);
    endp = Array{Float64,1}(undef, 3);
    for k = 1:3
        startp[k] = electrode.start_point[k];
        endp[k] = electrode.end_point[k];
    end
    increment = (endp - startp)/num_segments;
    for k = 0:num_segments
        nodes[k+1,:] = startp + k*increment;
    end
    segments = Array{Electrode,1}(undef, num_segments);
    for k = 1:num_segments
        segments[k] = new_electrode(nodes[k,:], nodes[k+1,:], electrode.radius,
                                    electrode.zi);
    end
    return segments, nodes
end;

function matchrow(a, B, atol=1e-9, rtol=0)
	#= Returns the row in B that matches a. If there is no match, nothing is returned.
	taken and modified from from https://stackoverflow.com/a/32740306/6152534 =#
	findfirst(i -> all(j -> isapprox(a[j], B[i,j], atol=atol, rtol=rtol), 1:size(B,2)), 1:size(B,1))
end

function seg_electrode_list(electrodes, frac)
	num_elec = 0; #after segmentation
	for i=1:length(electrodes)
		#TODO store in array to avoid repeated calculations
		num_elec += Int(ceil(electrodes[i].length/frac));
	end
	elecs = Array{Electrode}(undef, num_elec);
	nodes = zeros(Float64, (2*num_elec, 3));
	e = 1;
	nodes = [];
	for i=1:length(electrodes)
		ns = Int(ceil(electrodes[i].length/frac));
		new_elecs, new_nodes = segment_electrode(electrodes[i], ns);
		for k=1:ns
			elecs[e] = new_elecs[k];
			e += 1;
		end
		if (nodes == [])
			nodes = new_nodes;
		else
			for k=1:size(new_nodes)[1]
				if (matchrow(new_nodes[k:k,:], nodes) == nothing)
					nodes = cat(nodes, new_nodes[k:k,:], dims=1);
				end
			end
		end
	end
	return elecs, nodes
end;

function electrode_grid(a, n::Int, b, m::Int, h, r, zi=0.0im)
	#=
	Creates an electrode grid `h` coordinate below ground with each conductor
	having radius `r` and internal impedance `zi`.
	The grid has dimensions `a*b` with `n` and `m` divisions respectively.
	=#
	xx = 0:a/n:a;
	yy = 0:b/m:b;
	num_elec = n*(m + 1) + m*(n + 1);
	electrodes = Array{Electrode}(undef, num_elec);
	e = 1;
	for k=1:(m+1)
		for i=1:n
			electrodes[e] = new_electrode([xx[i], yy[k], h], [xx[i+1], yy[k], h], r, zi);
			e += 1;
		end
	end
	for k=1:(n+1)
		for i=1:m
			electrodes[e] = new_electrode([xx[k], yy[i], h], [xx[k], yy[i+1], h], r, zi);
			e += 1;
		end
	end
	# TODO return nodes as well?
	return electrodes
end;

function integrand_double(sender::Electrode, receiver::Electrode, gamma::ComplexF64, t)
	point_r = t[1]*(receiver.end_point - receiver.start_point) + receiver.start_point;
	point_s = t[2]*(sender.end_point - sender.start_point) + sender.start_point;
	r = norm(point_r - point_s);
	return exp(-gamma*r)/r;
end;

function integral(sender::Electrode, receiver::Electrode, gamma::ComplexF64,
	              maxevals=typemax(Int), atol=0, rtol=sqrt(eps(Float64)), norm=norm,
				  initdiv=1)
	#TODO other integration types; for now, only double is implemented in pure Julia
	f(t) = integrand_double(sender, receiver, gamma, t);
	intg, err = hcubature(f, [0., 0.], [1., 1.]; norm=norm, rtol=rtol, atol=atol,
			 maxevals=maxevals, initdiv=initdiv);
	return (intg*sender.length*receiver.length)
end;

function calculate_impedances(electrodes, gamma, s, mur, kappa, max_eval=typemax(Int),
                              req_abs_error=0, req_rel_error=sqrt(eps(Float64)),
							  error_norm=norm, initdiv=1)
	#TODO other integration types; for now, only double is implemented in pure Julia
	iwu_4pi = s*mur*MU0/(FOUR_PI);
    one_4pik = 1.0/(FOUR_PI*kappa);
    ns = length(electrodes);
    zl = zeros(Complex{Float64}, (ns,ns));
    zt = zeros(Complex{Float64}, (ns,ns));
	for i=1:ns
		ls = electrodes[i].length;
		k1 = electrodes[i].radius/ls;
		k2 = sqrt(1.0 + k1*k1);
		cost = 2.0*(log( (k2 + 1.)/k1 ) - k2 + k1);
		zl[i,i] = iwu_4pi*ls*cost + electrodes[i].zi;
		zt[i,i] = one_4pik/ls*cost;
		for k=(i+1):ns
            lr = electrodes[k].length;
            cost = 0.0;
            for m=1:3
                k1 = (electrodes[i].end_point[m] - electrodes[i].start_point[m]);
                k2 = (electrodes[k].end_point[m] - electrodes[k].start_point[m]);
                cost += k1*k2;
            end
            cost = abs(cost/(ls*lr));
            intg = integral(electrodes[i], electrodes[k], gamma, max_eval,
			                req_abs_error, req_rel_error,  error_norm, initdiv);
            zl[k,i] = iwu_4pi*intg*cost;
            zt[k,i] = one_4pik/(ls*lr)*intg;

            zl[i,k] = zl[k,i];
            zt[i,k] = zt[k,i];
        end
	end
    return zl, zt
end;

function impedances_images(electrodes, images, zl, zt, gamma, s, mur, kappa,
						   ref_l, ref_t, max_eval=typemax(Int),
						   req_abs_error=0, req_rel_error=sqrt(eps(Float64)),
						   error_norm=norm, initdiv=1)
	iwu_4pi = s*mur*MU0/(FOUR_PI);
    one_4pik = 1.0/(FOUR_PI*kappa);
    ns = length(electrodes);
	for i=1:ns
		ls = electrodes[i].length;
		for k=i:ns
            lr = images[k].length;
            cost = 0.0;
            for m=1:3
                k1 = (electrodes[i].end_point[m] - electrodes[i].start_point[m]);
                k2 = (images[k].end_point[m] - images[k].start_point[m]);
                cost += k1*k2;
            end
            cost = abs(cost/(ls*lr));
            intg = integral(electrodes[i], images[k], gamma, max_eval,
			                req_abs_error, req_rel_error,  error_norm, initdiv);
            zl[k,i] += ref_l*iwu_4pi*intg*cost;
            zt[k,i] += ref_t*one_4pik/(ls*lr)*intg;

            zl[i,k] = zl[k,i];
            zt[i,k] = zt[k,i];
		end
	end
    return zl, zt
end;

function incidence(electrodes::Vector{Electrode}, nodes::Matrix{Float64})
	# build incidence matrices for calculating 'YN = AT*inv(zt)*A + BT*inv(zl)*B'
    ns = length(electrodes);
    nn = size(nodes)[1];
    a = zeros(Float64, (ns,nn));
    b = zeros(Float64, (ns,nn));
    for i = 1:ns
        for k = 1:nn
            if isapprox(collect(electrodes[i].start_point), nodes[k,:])
                a[i,k] = 0.5;
                b[i,k] = 1.0;
            elseif isapprox(collect(electrodes[i].end_point), nodes[k,:])
                a[i,k] = 0.5;
                b[i,k] = -1.0;
            end
        end
    end
    return a, b
end;
