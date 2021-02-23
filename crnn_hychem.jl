include("header.jl")

is_restart = false;
n_epoch = 100000;
ntotal = 20;
batch_size = 16;
n_plot = 2;
grad_max = 10.0^(-1);
maxiters = 1000;
n_exp = 10;
noise = 0.01;
nr = 6;
opt = ADAMW(0.001, (0.9, 0.999), 1.e-6);
atol = 1.e-8;
rtol = 1.e-3;
lb = atol;
ub = 1.0;

include("dataset.jl")
include("network.jl")
include("callback.jl")
# opt = ADAMW(5.e-3, (0.9, 0.999), 1.f-6)
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        sample = rand(batch_size:ntotal)
        loss_epoch[i_exp], ind = loss_n_ode(p, i_exp, ntotal; get_ind=true)
        grad = ForwardDiff.gradient(
            x -> loss_n_ode(x, i_exp, minimum([ind, sample])),
            p,
        )
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    _loss = mean(loss_epoch)
    _gnorm = mean(grad_norm)
    set_description(
        epochs,
        string(
            @sprintf("Loss %.4e grad %.2e lr %.2e", _loss, _gnorm, opt[1].eta)
        ),
    )
    cb(p, _loss, _gnorm)
end
