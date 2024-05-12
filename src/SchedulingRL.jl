module SchedulingRL

using StatsBase
using Random
using JSON
using Plots
using ColorSchemes
using GraphRecipes
using Distributions
using SparseArrays
using VegaLite
using DataFrames
using LinearAlgebra

using Flux
using OptimizationOptimisers
using ParameterSchedulers
using ProgressMeter
using Functors: @functor
using ChainRulesCore
using CircularArrayBuffers

# packages for outputs?
using PrettyTables
using Printf
using Latexify


# to hook outputs to TensorBoard
using TensorBoardLogger, Logging
using TimeZones 
# Write your package code here.
include("Sim/Environment.jl")
include("Agent.jl")

"""
    testing(experiment::Experiment)
"""
function testing()

end

export  createenv, setsamples!, statesize, actionsize,
        createGP, createE2E, createWPF, ActorCritic, GaussianNetwork, Priorityfunction, numberweights,
        createagent, updateagentsettings!, trainagent!, exportagent, showsettings, setenvs!, setobj!,
        testagent, creategantt, generateset, 
        createRule, createExpSequence, createRandomAction,
        createGES,
        build_tree, drawTree, PriorityFunctionFromTree, analyze_tree, map_features_to_primitives, get_expression, simplify_expression,reduceTreeNumericals, Priorityfunction,
        setinitalparameters, evalfitnesssample, testfitness, Covert_weights_from_GP, testGES,actionsfromindividuum,resetenv!,numberweights,
        save_weights_boxplot, save_results_diagram, save_objective_evolution, latexify_dispatching_rules, latexify
        # add more visualization methods for results here?
        # and modes? like real time visualization of scheduling -> real time Gantt :D



end
