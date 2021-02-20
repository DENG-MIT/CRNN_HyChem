gas = CreateSolution("./mechanism/JP10skeletal.yaml")
const MW = gas.MW

varnames = [
    "C10H16",
    "H",
    "H2",
    "CH3",
    "CH4",
    "aC3H5",
    "C2H4",
    "C3H6",
    "C5H6",
    "C6H6",
    "C6H5CH3",
    "OH",
    "HO2",
    "H2O2",
    "O",
    "O2",
    "H2O",
];

E_C = [10, 0, 0, 1, 1, 3, 2, 3, 5, 6, 7, 0, 0, 0, 0, 0, 0];
E_H = [16, 1, 2, 3, 4, 5, 4, 6, 6, 6, 8, 1, 1, 2, 0, 0, 2];
E_O = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1];
E_ = hcat(E_C, E_H)[:, :];
E_null = nullspace(E_')';
nse = size(E_null)[1];

varnames_obs = [
    "C10H16",
    "H2",
    "CH4",
    "C2H4",
    "C3H6",
    "C5H6",
    "C6H6",
    "C6H5CH3",
    "O2",
    "H2O",
]

ind_crnn = []
for var in varnames
    push!(ind_crnn, species_index(gas, var))
end

ind_obs = []
for var in varnames_obs
    push!(ind_obs, species_index(gas, var))
end

ind_N2 = species_index(gas, "N2")
ns = length(ind_obs);

yall = zeros(n_exp, ns + 3, ntotal);
for i_exp = 1:n_exp
    rawdata = readdlm("data/data_$i_exp")'
    ts = rawdata[1, :]
    tend = ts[end]

    _ts = 10 .^ range(log10(tend / 1e4), log10(tend / 1.01), length = ntotal)
    _ts[1] = 0.0

    _ylabel = zeros(ns + 2, ntotal)
    for i = 1:ns+2
        itp = LinearInterpolation(ts, rawdata[i+1, :])
        _ylabel[i, :] .= itp.(_ts)
    end

    yall[i_exp, 1, :] .= _ts
    yall[i_exp, 2:end, :] .= _ylabel
end
yalls = yall[:, 4:end, :];
# yscale = (maximum(yalls, dims=[1, 3]) - minimum(yalls, dims=[1, 3]) .+ lb)[1, :, 1];

println("include ", @__FILE__)
