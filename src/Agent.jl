abstract type AbstractApproximator end
"""
mutable struct of an Agent 
"""
mutable struct Agent{A<:AbstractApproximator}
    "defines the type of the agent, i.e. GP, DRL, or ParaPrio"
    type::String
    "action space the agent uses to train"
    actionspace::String
    "mask that is used additionally"
    actionmask::Array{String}
    "task the agent is trained, i.e. tailored or generalized"
    purpose::String
    "will store the problem it was assigned to as string"
    problemInstance # placeholder: problem was solved
    "will store the envs used for training"
    trainingenvs::Union{Vector{Env},Nothing} # placeholder: problem was solved
    "objective the agent is trained for,"
    "[makespan, totalFlowTime, tardiness, discretetardiness, numberTardyJobs, numSetups]" #TODO length of input assert??
    objective::Vector{Float64}
    "stores the struct of the intialized agent based on previous inputs"
    approximator::A # initialized structure for the agent based on settings
    "stores the trained model"
    model # placeholder: model
    checkpoint # stores checkpoints in form [model, value]
    rng::AbstractRNG
    logger::Dict{String,Vector{Any}}
    
end

include("Approximators/Approximators.jl")

"""
    create_agent(type::String, actionspace:: String; purpose::String = "tailored", obj = [1.0,0.0,0.0,0.0,0.0,0.0], settings::Dict = Dict())

creates the agent and returns it as a struct
"""
function createagent(approximator::AbstractApproximator, actionspace::String; actionmask = ["none"], purpose::String = "tailored",
    weight_makespan = 1.0, weight_totalFlowTime = 0.0, weight_tardiness = 0.0, weight_discretetardiness = 0.0, weight_numberTardyJobs = 0.0, weight_numSetups = 0.0)
    
    obj = [weight_makespan, weight_totalFlowTime, weight_tardiness, weight_discretetardiness, weight_numberTardyJobs, weight_numSetups]
    if approximator.trainable
        setobj!(approximator, obj)
        envs = approximator.envs # TODO
        instancename = envs[1].name
    else
        envs = nothing
        instancename = nothing
    end
    logging = Dict("reward" => [])
    type = string(typeof(approximator))[14:end]

    Agent(type, actionspace, actionmask, purpose, instancename , envs, obj, approximator, nothing, [],
    type == "E2E" || type == "WPF" ? approximator.policy.rng : approximator.rng, logging)
end

"""
    update_agent_settings!(agent::Agent; settings::Dict = Dict(), obj= [1.0,0.0,0.0,0.0,0.0,0.0])

    change settings or objective of agent
    resets model / might be updated to keep model -> test effect of keeping model for other purposes

"""
function updateagentsettings!(agent::Agent; settings::Dict = Dict(), obj= [1.0,0.0,0.0,0.0,0.0,0.0])

    for c in keys(settings)
        if haskey(agent.settings,c)
            agent.settings[c] = settings[c]
        end
    end

    agent.objective = obj
    agent.approximator = initAgent(agent.type, agent.trainingenvs, obj, agent.settings)
    
    # TODO store intermediate result! change in order to keep model and only train further?
    agent.model = nothing

    println("Agent's settings are updated")
end

"""
   create_agent(file::String)

create agent instead from a file -> trained model
"""
function createagent(file::String)
    # TODO
    return Agent
end

"""
    trainagent!(agent::Agent; generations=100, evalevery = 0, finaleval = true, kwargs...)

    evalevery: how often to evaluate the agent, 0 => no eval while training

traines the agent and changes its model variable  
"""
function trainagent!(agent::Agent; generations=100, validation=false, validate_every=0, validationenvs=nothing, kwargs...)
    # training loop call function based on type of agent / i.e. approximator
    if validation || validate_every > 0
        @assert validationenvs !== nothing "No envs/samples for validation provided."
    end
    @assert length(agent.trainingenvs) > 0 "No environment set to train on. call setenvs! on Agent first"
    train!(agent, generations; validationenvs=validationenvs, validation=validation, validate_every=validate_every, kwargs...)
end

"""
    export_agent(agent::Agent, target::String; kwargs)

exports the agent and allows it to be stored in the target location  
"""
function exportagent(agent::Agent, name::String, target::String; kwargs...)

end

"""
    show_settings(agent::Agent)

    prints the settigns of the agent to the console  
"""
function showsettings(agent::Agent)
    if agent.model !== nothing
        s = "(trained)"
    else
        s ="(untrained)"
    end
    
    @printf("\n Agent %s general settings are: ", s)
    println()

    df_stats = DataFrame(Setting=String[],Value=Any[])
    push!(df_stats, ("Type", agent.type))
    push!(df_stats, ("Action Space", agent.actionspace))
    push!(df_stats, ("Purpose", agent.purpose))
    push!(df_stats, ("Objective", agent.objective))
    push!(df_stats, ("problemInstance", agent.problemInstance))
    push!(df_stats, ("different envs ", length(agent.trainingenvs)))
    pretty_table(df_stats,alignment=:c)
    println("Agent specific settings are:")
    pretty_table(agent.settings,alignment=:c)
end

"""
    setenvs!(agent::Agent, envs::Vector{Env})

    set the env of an agent to train on
    if existing already replaces them
    propagated to algorithm envs
"""
function setenvs!(agent::Agent, envs::Vector{Env})
    for env in envs
        env.actionspace = agent.actionspace
    end
    # TODO for NN based agents add check if apdding is active or not and statesize/actionsize vs env required
    agent.trainingenvs = envs
    if length(agent.approximator.envs) >0
        for _ in eachindex(agent.approximator.envs)
            deleteat!(agent.approximator.envs,1)
        end
        append!(agent.approximator.envs,envs)
    else
        append!(agent.approximator.envs,envs)
    end
end

function setobj!(agent::Agent, obj::Vector{Float64})
    agent.objective = obj
    if length(agent.approximator.objective) >0
        for _ in eachindex(agent.approximator.objective)
            deleteat!(agent.approximator.objective,1)
        end
        append!(agent.approximator.objective,obj)
    else
        append!(agent.approximator.objective,obj)
    end
end

function generateset(envstring::Vector{Dict{String,String}}; rng = Random.default_rng(), fromfile::Bool=true, n_timesamples::Int=100, randomorderbook::Bool=false,
                    n_orderbookssamples::Int=30, nOrders::Int=40, actionspace = "AIM", actionmask_sorting = ["none"], deterministic = false)
    @assert deterministic == false || n_timesamples == 1 "deterministic only requries one time sample"
    @assert fromfile == false || randomorderbook == false "random orderbook not supported with fromfile"
    @assert fromfile == false || n_timesamples <= 100 "fromfile only supports up to 100 sample"

    envs::Vector{Env} = []
    for e in envstring
        if randomorderbook
            for _ in 1:n_orderbookssamples
                push!(envs, createenv(fromfile=false, randomorderbook=true, nOrders=nOrders, actionspace=actionspace, actionmask_sorting=actionmask_sorting,
                                    nSamples=n_timesamples, layout = e["layout"] ,instanceType = e["instancetype"],
                                    instanceName = e["instancename"], deterministic = deterministic
                ))
            end
        else
            tmp_env = createenv(fromfile=fromfile, actionspace=actionspace, actionmask_sorting=actionmask_sorting, nSamples=n_timesamples,
                                layout=e["layout"] ,instanceType=e["instancetype"],
                                instanceName=e["instancename"], deterministic=deterministic
                                )
            if !fromfile
                setsamples!(tmp_env, rng, nSamples = n_timesamples, antithetic = false)
            end
            push!(envs, tmp_env)
        end
    end
    return envs
end


"""
    testagent(agent::Agent,envs::Vector{String},nrseeds)


    test trained agent with envs. each env needs to be created with the samples to evaluate!
"""
function testagent(agent::Agent, testenvs) #TODO or only add Json files and envs are generated within function

    # absolute metrics
    totalobjVal = 0
    envobj = []
    envseedobj = []
    if agent.type == "GP"
        insamplevalue = agent.model[1][2]
    elseif agent.type == "ExpSequence"
        insamplevalue = agent.approximator.expsol
    else
        insamplevalue = 1.0 # TODO 
    end
    nrseeds = max(length(testenvs[1].samples),1)
    for env in testenvs
        tmpO,tmpS = test(agent,env,nrseeds) # returns total env objective, objective for each seed and the gap of each seed
        totalobjVal += tmpO
        tmpenvobj = tmpO/nrseeds
        append!(envobj,tmpenvobj)
        append!(envseedobj,tmpS)
    end
    totalobjVal /= (nrseeds*length(testenvs))
    losstoinsample = (insamplevalue/totalobjVal)-1
    return totalobjVal, envobj, envseedobj, losstoinsample
end


function creategantt(agent::Agent,envstring,seed; env=nothing, deterministic::Bool = false)
    if env === nothing
        env = createenv(fromfile = true, specificseed = seed, as = agent.actionspace, am = agent.actionmask, nSamples = 1,
                                layout = envstring["layout"] ,instanceType = envstring["instancetype"],
                                instancesettings = envstring["instancesettings"],
                                datatype = envstring["datatype"] ,instanceName = envstring["instancename"])
    else
        resetenv!(env)
    end
    assignments = DataFrames.DataFrame()
    sequence = Dict(m => [] for m in env.problem.machines) #TODO use / store? or move to normal testing and store all and compare?
    # TODO define individuum, store intraiing performance as well!
    t = false
    if deterministic == false
        setsamplepointer!(env.siminfo,1)
    end
    while !t
        action = nextaction(agent,env)
        if action[3] == "R"
            setuptime = deterministic ? env.problem.setupstatejob[action[2]][env.state.setup[action[2]]][action[1]]["mean"] : env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]]
            append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string("Setup ",action[1]),
            Start = env.siminfo.time, Machine = action[2],
            End = (env.siminfo.time + setuptime)))

        elseif action[3] == "S"
            push!(sequence[action[2]],action[1])
            operation = env.problem.orders[action[1]].ops[env.siminfo.pointercurrentop[action[1]]]
            processingtime = deterministic ? env.problem.orders[action[1]].processingtimes[operation][action[2]]["mean"] : env.samples[1]["processing_time"][action[1]][operation][action[2]]
            if env.state.setup[action[2]] === nothing || env.problem.setupstatejob[action[2]][env.state.setup[action[2]]][action[1]]["mean"] == 0
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string(action[1]),
                Start = env.siminfo.time, Machine = action[2],
                End = (env.siminfo.time + processingtime)))
            else
                setup_time = deterministic ? env.problem.setupstatejob[action[2]][env.state.setup[action[2]]][action[1]]["mean"] : env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]]
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string("Setup ",action[1]),
                Start = env.siminfo.time, Machine = action[2],
                End = (env.siminfo.time + setup_time)))
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string(action[1]),
                Start = (env.siminfo.time + setup_time), Machine = action[2],
                End = (env.siminfo.time + setup_time + processingtime)))
            end
        end
        state, reward, t, info = step!(env,action, agent.rng)
    end
    # setupnames = [string("Setup ", x) for x in keys(env.problem.orders)]
    # processnames = [string(x) for x in keys(env.problem.orders)]
    # colDomain = [setupnames, processnames]
    # colRange = ["#e08c8c", "#dc0202", "#fff998", "#fff000", "#80e78e", "#00ad18", "#85a5ee", "#004dff"]
   
    p = assignments |> @vlplot(
        mark = {
            type = :bar,
            cornerRadius = 10
        },
        width = 2000,
        height = 1000,
        y="Machine:n",
        x="Start:q",
        x2="End:q",
        color={
            field="Type",
            type = "nominal",
            #scale = {domain = colDomain, range = colRange}
        } 
    ) 
    +
    @vlplot(
        mark={:text,align = "center",x="Start", y="Machine", dx=20, dy=10},
        text="Job",
        detail={aggregate="count",type="qualitative"}
    )
    # + 
    # @vlplot(
    #     mark={:text,align = "center",x="Start", y="Machine", dy=-10},
    #     text="Due"
    # )
    # VegaLite.save(string(@__DIR__,"/Plots/",instanceName,"_",SIM_METHOD, "_gantt.pdf"), p)
    display(p)
end