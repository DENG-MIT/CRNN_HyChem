cbi = function (p, i_exp)
    pred = predict_n_ode(p, i_exp)[ind_obs, :] .+ lb
    ts = @views(yall[i_exp, 1, :])[1:size(pred)[2]]
    ylabel = @views(yall[i_exp, 4:end, 1:size(pred)[2]])

    list_plt = []
    for i in 1:ns
        plt = scatter(ts .+ lb, ylabel[i, :], xscale=:log10, label="data")
        plot!(plt, ts .+ lb, pred[i, :], lw=3, xscale=:log10, label="pred")
        title!(plt, "$(varnames_obs[i])")
        # xlabel!(plt, "Time [s]")
        push!(list_plt, plt)
    end
    plt = plot(ts .+ lb, sum(ylabel, dims=1)' .- 1, lw=3, xscale=:log10, label="data")
    plot!(plt, ts .+ lb, sum(pred, dims=1)' .- 1, lw=3, xscale=:log10, label="pred")
    title!(plt, "Y_sum - 1")
    push!(list_plt, plt)
    plt_all = plot(list_plt..., legend=false, size=(1000, 1000))
    png(plt_all, "figs/pred-$i_exp.png")
end

list_loss = [];
list_grad = [];
iter = 1;
cb = function (p, loss_mean, g_norm; doplot=true)
    global list_loss, list_grad, iter
    push!(list_loss, loss_mean)
    push!(list_grad, g_norm)

    if doplot & (iter % n_plot == 0)
        display_p(p)

        i_exp = randperm(n_exp)[1]
        cbi(p, i_exp)
        println("plot $i_exp")

        plt_loss = plot(list_loss, yscale=:log10, label="loss");
        plt_grad = plot(list_grad, yscale=:log10, label="grad_norm");
        xlabel!(plt_loss, "Epoch");
        xlabel!(plt_grad, "Epoch");
        ylabel!(plt_loss, "Loss");
        ylabel!(plt_grad, "Grad Norm");
        plt_all = plot([plt_loss, plt_grad]..., legend=:bottomleft);
        png(plt_all, "figs/loss_grad");

        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
    return false
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
end
