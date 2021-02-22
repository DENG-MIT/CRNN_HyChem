np = nr * (nse + 3) + 1;
p = randn(Float64, np) .* 0.1 .- 0.05;
p[1:nr] .= 0.1;  #lnA
p[nr * 2 + 1:nr * 3] .= -0.1;  #Ea
p[end] = 0.1;  # slope
@inbounds function p2vec(p)
    slope = p[end] .* 100.0
    w_b = p[1:nr] .* slope
    w_in_b = p[nr + 1:nr * 2]
    w_in_Ea = p[nr * 2 + 1:nr * 3] .* slope
    w_out = E_null' * reshape(p[nr * 3 + 1:nr * (nse + 3)], nse, nr)
    w_in = vcat(clamp.(-w_out, 0.0, 4.0), w_in_Ea', w_in_b')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p)
    println("\n species (row) reaction (col)")
    println(varnames[1:8])
    println(varnames[9:end])
    println("w_out | Ea | b | logA")
    show(stdout, "text/plain", round.(hcat(w_out', w_in'[:, end - 1:end], w_b)', digits=2))
    # println("\n w_out")
    # show(stdout, "text/plain", round.(w_out', digits = 2))
    println("\n w_nse")
    w_e = reshape(p[nr * 3 + 1:nr * (nse + 3)], nse, nr)
    show(stdout, "text/plain", round.(w_e, digits=2))
    println("\n")
end
display_p(p)

@inbounds function crnn!(du, u, p, t)
    T::typeof(u[1]) = itpT(t)
    P = itpP(t)
    Y = clamp.(u, -lb, 1.0)
    mean_MW = 1.0 / dot(Y, 1 ./ MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)
    wdot = wdot_func(gas.reaction, T, C, S0, h_mole)
    crnn_in =
        vcat(log.(clamp.(@view(C[ind_crnn]), lb, 100.0)), -1.0 / T, log(T))
    wdot[ind_crnn] .+= w_out * exp.(w_in' * crnn_in + w_b)
    @. du = wdot * MW / ρ_mass
end

tspan = [0, 0.01];
u0 = zeros(gas.n_species);
prob = ODEProblem(crnn!, u0, tspan)

ode_solver = TRBDF2();
sense = ForwardDiffSensitivity()
@inbounds function predict_n_ode(p, i_exp, sample=ntotal; dense=false)
    global w_in, w_b, w_out = p2vec(p)
    ylabel = @view(yall[i_exp, :, :])
    ts = @view(ylabel[1, :])
    global itpT =
        LinearInterpolation(ts, @view(ylabel[2, :]); extrapolation_bc=Flat())
    global itpP =
        LinearInterpolation(ts, @view(ylabel[3, :]); extrapolation_bc=Flat())
    u0 = zeros(gas.n_species)
    u0[ind_obs] .= @view(ylabel[4:end, 1])
    u0[ind_N2] = 1 - sum(u0)
    _prob = remake(prob, u0=u0, p=p, tspan=[0, ts[sample]])
    if dense
        _ts = []
    else
        _ts = @view(ts[1:sample])
    end
    sol = solve(
        _prob,
        ode_solver,
        saveat=_ts,
        atol=atol,
        rtol=rtol,
        sensalg=sense,
        maxiters=maxiters,
    )
    pred = Array(sol)
end
@time pred = predict_n_ode(p, 1; dense=false);

@inbounds function loss_n_ode(p, i_exp, sample=ntotal; get_ind=false)
    pred = @views(clamp.(predict_n_ode(p, i_exp, sample), -ub, ub)[ind_obs, :])
    ind = size(pred)[2]

    if get_ind & (minimum(pred[1, :]) < lb)
        ind = findfirst(pred[1, :] .< 2 * lb)
        ind = maximum([2, ind])
    end
    pred = @view(pred[:, 1:ind])

    ylabel = clamp.(@view(yall[i_exp, 4:end, 1:ind]), -ub, ub)
    # yscale_ = maximum(ylabel, dims=2) - minimum(ylabel, dims=2) .+ 1.e-6
    loss = mae(pred, ylabel)

    if get_ind
        return loss, ind
    else
        return loss
    end
end
@time loss_n_ode(p, 1; get_ind=true)
