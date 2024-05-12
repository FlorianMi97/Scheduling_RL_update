
const DATA_DIR = "/data"

abstract type AbstractEnv end
include("Problem.jl")
include("States.jl")
include("Siminfo.jl")

"""
    mutable struct for one problem
        problem,initialState are immutable
        samples are mutable (change during training or in DRL constantly changing?)
        state is mutable to execute SIM

    if considering multiple problems => create multiple envs
"""
Base.@kwdef mutable struct Env <: AbstractEnv
    # immutable
    problem::Problem
    actionspace::String
    actionmasks::Array{String}
    initialState:: InitialState
    type::String
    name::String
    deterministic::Bool

    # mutable
    state::State
    siminfo::Siminfo
    nrsamples::Int = 30
    samples::Dict

end

include("Simulation.jl")
include("Actionmask.jl")

"""
    createenv(;fromfile::Bool = false , specificseed = nothing, layout::String = "Flow_shops" ,instanceType::String = "doerr",
                ,instanceName::String = "default_problem", actionspace::String = "AIM", actionmask_sorting::Array{String} = [""],
                nSamples = 30, deterministic::Bool = false, kwargs...)

    creates the enviroment based on the input file.  
    return a env struct
"""
function createenv(;fromfile::Bool=false, randomorderbook::Bool=false, nOrders::Int = 40, specificseed=nothing, layout::String="Flow_shops" ,instanceType::String="doerr",
                        instanceName::String="default_problem", actionspace::String="AIM", actionmask_sorting::Array{String}=["none"], rng = Random.default_rng(),
                        nSamples=30, deterministic::Bool=false, kwargs...)

    @assert actionspace in ["AIM", "AIA", "AIAR"] "actionspace not supported"
    @assert instanceType in ["usecase", "kurz", "doerr"] "instanceType not supported"
    @assert all(x -> x in ["dds", "ois", "none"], actionmask_sorting) "actionmask_sorting not supported"
    @assert !(("dds" in actionmask_sorting) && ("ois" in actionmask_sorting)) "either dds or ois can be used"
    @assert randomorderbook == false || instanceType == "doerr" "randomorderbook only supported for doerr instances"
    @assert randomorderbook == false || fromfile == false "random orderbook not supported with fromfile"

    dir =  pathof(SchedulingRL)|> dirname |> dirname

    # create problems with different readers based on input data/type of instances
    if instanceType == "usecase"
        problemfile = string(dir,"/data/",layout, "/Use_Case_Cereals/",instanceName,".json")
    elseif instanceType == "kurz"
        problemfile = string(dir,"/data/",layout, "/Benchmark_Kurz/", instanceName, ".json")
    elseif instanceType == "doerr"
        problemfile = string(dir,"/data/",layout, "/Benchmark_Doerr/Instances/", instanceName, ".json")
    end

    # Problem(file; id = 1, randomOrderbooks = false, mOrders = 55, actPro = ["CEREAL_1", "CEREAL_2", "CEREAL_3"], Seed = nothing, kwargs...)
    p = Problem(layout, instanceType, problemfile, randomorderbook = randomorderbook, nOrders = nOrders, rng = rng, kwargs...)
    # create intial state
    i = InitialState(p, actionspace, actionmask_sorting)
    # create state (based on initial state)
    state = State(i)
    info = initsiminfo(p, actionmask_sorting)
    s = fromfile ? readuncertainities(p, nSamples, layout, instanceType, instanceName, specificseed) : Dict() # empty dict. needs to be set!
    Env(p, actionspace, actionmask_sorting,i,instanceType,instanceName, deterministic, state, info, nSamples, s)
end

"""
    used to create env based on a Json file and a folder with JSON files for the realizations of uncertainties based on seeds!
"""
function readuncertainities(problem::Problem, nSamples::Int, layout::String, instanceType::String, instanceName::String, specificseed)
    dir =  pathof(SchedulingRL) |> dirname |> dirname

    if instanceType == "usecase"
        foldername = string(dir,"/data/",layout, "/Use_Case_Cereals/testing/",instanceName)
    elseif instanceType == "kurz"
        foldername = string(dir,"/data/",layout, "/Benchmark_Kurz/testing/", instanceName)
    elseif instanceType == "doerr"
        foldername = string(dir,"/data/",layout, "/Benchmark_Doerr/testing/", instanceName)
    end
    data = Dict()
    if specificseed === nothing
        filesinfolder = readdir(foldername)
        @assert nSamples <= length(filesinfolder) "too many samples selected (should be less than 100 later on)"
        for (i,filename) in enumerate(filesinfolder[1:nSamples])
            filepath = string(foldername,"/",filename)
            open(filepath, "r") do f
                data[i] = changekeys(problem,JSON.parse(f),instanceType)
            end
        end
    else
        filepath = string(foldername,"/",specificseed,".json")
            open(filepath, "r") do f
                data[1] = changekeys(problem,JSON.parse(f),instanceType)
            end
    end

    return data
end

function changekeys(problem::Problem, sampleDict, instanceType)

    prodict = Dict(j => Dict(o => Dict() for o in problem.ops_per_prod[problem.orders[j].type]) 
                                         for j in keys(problem.orders))                                   
    setdict = Dict(m => Dict(pr => Dict() for pr in vcat(problem.prods,[nothing]))
                                        for m in problem.machines) 
    sample_dict = Dict("processing_time" => prodict, "setup_time" => setdict, "seed" => sampleDict["seed"])
    
    for j in keys(problem.orders)
        job = problem.orders[j]
        for o in job.ops
            for m in job.eligmachines[o]
                sample_dict["processing_time"][j][o][m] = sampleDict["processing_time"][problem.mapindexorder[j]][problem.mapindexops[o]][problem.mapindexmachine[m]]

                #sample setup time (dependent on predecessor)
                for pr in problem.prods #setdiff(problem.prods, [prod])
                    sample_dict["setup_time"][m][pr][j] = sampleDict["setup_time"][problem.mapindexmachine[m]][problem.mapindexprods[pr]][problem.mapindexorder[j]]
                end
                sample_dict["setup_time"][m][nothing][j] = sampleDict["initial_setup"][problem.mapindexmachine[m]][problem.mapindexorder[j]]   
            end
        end

    end
    sample_dict
end

"""
reset the env to its initial state, i.e. state and siminfo are resetted!
"""
function resetenv!(env::Env)
    #create samples s
    env.siminfo = initsiminfo(env.problem, env.actionmasks)
    env.state = State(env.initialState)
end

"""
changes the samples in the env. Allows to constantly resample.
or change seeds for experiments
"""
function setsamples!(env::Env, rng::AbstractRNG; nSamples = 30, antithetic = false)
    #create samples s
    env.samples = antithetic ? sampleuncertainitiesantithetic(env.problem, nSamples, rng) : sampleuncertainties(env.problem, nSamples, rng)
end
