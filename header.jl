using OrdinaryDiffEq, Flux, Plots
using DiffEqSensitivity
using ForwardDiff
using Interpolations
using LinearAlgebra
using Random
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae
using Interpolations:Flat
using BSON: @save, @load
using DelimitedFiles
using Arrhenius

ENV["GKSwstype"] = "100"

if ispath("figs") == false
    mkdir("figs")
end

if ispath("checkpoint") == false
    mkdir("checkpoint")
end
