Base.@kwdef mutable struct GES <: AbstractApproximator
    
    granularity::String
    envs::Vector{Env}
    objective::Vector{Any}
    priorityfunctions::Vector{Priorityfunction}
    bestweights::Vector{Vector{Float32}}
    weights_with_signs::Vector{Vector{Float32}}
    sigma::Vector{Vector{Float64}}
    rng::AbstractRNG
    trainable::Bool = true
  

    
    popSize::Int =48
    stages::Int =1

    #for weight Optimization
    optimizer::String="GradientDecent"
    #SGD
    learningrates_SGD::Vector{Float64} = [0.1, 0.025]
    learningrates_decay_SGD::Vector{Float64} = [0.99, 0.999]
    decay_SGD::Bool = false
    decayStepSize::Int = 1
    SGD_params::Dict{String, Any} = Dict("βμ" => 0.95, "βσ" => 0.95)
    nesterov_state::Dict{String, Any} = Dict()
    #ADAM
    learningrates_ADAM::Vector{Float64} = [0.05, 0.01]
    learningrates_decay_ADAM::Vector{Float64} = [0.95, 0.99]
    decay_ADAM::Bool = false
    adam_params::Dict{String, Any} = Dict("β1" => 0.9, "β2" => 0.99, "ϵ" => 1e-8)
    adam_state::Dict{String, Any} = Dict()

    # for pareto via marker density
    markerdensity::Bool = false
    markerstart::Int = 1
    markerend::Int = 3

    # mutable progress
    generation::Int = 0

    #sampling
    stationary_samples::Bool = true # if false change samples each x generations
    change_every::Int = 10 # if stationary_samples = false, change samples every x generations
    sample_cycling::Bool = false # if true cycle through samples, otherwise set new ones
    steps_in_cycle::Int = 10 # if sample_cycling = true, x cycles before taking first again
    sample_tens_digit::Int = 0 # if sample_cycling = true, pointer to start of current sample range
    samples_generation::Int = 60 # number of samples evaluated per generation
  
end

function GES(granularity::String, envs::Vector{Env}, priorityfunctions::Union{Priorityfunction, Vector{Priorityfunction}}, objective::Vector{Any}, rng::AbstractRNG; kwargs...)
    if granularity == "global"    
        stages = 1  
        bestweights = [fill(1.0, numberweights(priorityfunctions[1]))]
        weights_with_signs = [fill(1.0, numberweights(priorityfunctions[1]))]
        sigma = [fill(log(3), numberweights(priorityfunctions[1]))]

    
        return GES(granularity=granularity, envs=envs, objective=objective, priorityfunctions=priorityfunctions, bestweights=bestweights, weights_with_signs = weights_with_signs, rng=rng, sigma=sigma, stages=stages; kwargs...)
    
    elseif granularity == "stage"
        
        error("Stage granularity not supported yet")
        if priorityfunctions isa Priorityfunction
            error("Priorityfunctions don't match granularity")
        end
    
    else
        error("Granularity not supported")
        if priorityfunctions isa Priorityfunction
            error("Priorityfunctions don't match granularity")
        end
    end
end

 
function createGES(granularity::String, envs::Vector{Env}, priorityfunctions::Union{Priorityfunction, Vector{Priorityfunction}}, obj::Vector{Any}=[], rng::AbstractRNG=Random.default_rng(); kwargs...)

    # Handling the `priorityfunctions` to ensure it's always a Vector{Priorityfunction}
    priorityfunctions = priorityfunctions isa Priorityfunction ? [priorityfunctions] : priorityfunctions

    GES(granularity, envs, priorityfunctions, obj, rng; kwargs...)
end


function setinitialparameters(ges::GES; kwargs...)
    for (param, value) in kwargs
        if !isnothing(value)
            # Check and update for initialweights
            if param == :initialweights
                !(value isa Vector{Vector{Float32}}) && error("Initialweights must be of type Vector{Vector{Float32}} but are of type: ", typeof(value))
                ges.bestweights = value

            # Check and update for sigma
            elseif param == :sigma
                !(value isa Vector{Vector{Float32}}) && error("Sigma must be of type Vector{Vector{Float32}} but are of type: ", typeof(value))
                ges.sigma = value

            # Check and update for learningrates
            elseif param == :learningrates
                !(value isa Vector{Float32}) && error("Learningrates must be of type Vector{Float32} but are of type: ", typeof(value))
                ges.learningrates = value

            # Check and update for optimizer
            elseif param == :optimizer
                !(value isa String) && error("Optimizer not supported")
                ges.optimizer = value

            # Check and update for popsize
            elseif param == :popsize
                !(value isa Int) && error("Popsize must be of type Int")
                ges.popSize = value

            # Check and update for samples_generation
            elseif param == :samples_generation
                !(value isa Int) && error("Samples_generation must be of type Int but are of type: ", typeof(value))
                ges.samples_generation = value

            # Check and update for samples_generation
            elseif param == :stages
                !(value isa Int) && error("Stages must be of type Int but are of type: ", typeof(value))
                ges.stages = value

            else
                error("Parameter $param not recognized.")
            end
        end
    end
end