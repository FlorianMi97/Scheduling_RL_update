mutable struct ExpSequence <: AbstractApproximator
    sequence::Dict
    counterdict::Dict
    trainable::Bool
    rng::AbstractRNG
    actionspace::String
    expsol::Float64
end

function createExpSequence(instance, env; actionspace="AIA", exp_measure="mean", objective::String="makespan", rng=Random.default_rng())
    if actionspace == "AIM" @warn("AIM is not compatible with Exp. Sequence. This might cause errors") end
    exp_value_sol = Dict()
    dir =  pathof(SchedulingRL)|> dirname |> dirname
    @assert objective in ["tardiness", "makespan"] "objective must be either tardiness or makespan"

    if instance[1:3] == "dpg"
        if actionspace == "AIAR"
            file = string(dir,"/results/Flow_shops/Benchmark_Doerr/tailored/", instance[1:end-4], "/",
            objective, "/decoupled_setups/", "seq_", exp_measure, "_", instance, ".json")
        else
            file = string(dir,"/results/Flow_shops/Benchmark_Doerr/tailored/", instance[1:end-4], "/",
            objective, "/coupled_setups/", "seq_", exp_measure, "_", instance, ".json")
        end

    elseif contains(instance, "useCase")
        if actionspace == "AIAR"
            file = string(dir,"/results/Flow_shops/Use_Case_Cereals/decoupled_setups/seq_", instance, "_", objective, ".json")
        else
            file = string(dir,"/results/Flow_shops/Use_Case_Cereals/coupled_setups/seq_", instance, "_", objective, ".json")
        end

    else
        if actionspace == "AIAR"
            file = string(dir,"/results/Flow_shops/Benchmark_Kurz/decoupled_setups/seq_", instance, "_", objective, ".json")
        else
            file = string(dir,"/results/Flow_shops/Benchmark_Kurz/coupled_setups/seq_", instance, "_", objective, ".json")
        end
    
    end
    open(file, "r") do f
        exp_value_sol = JSON.parse(f)
    end
    sequence = exp_value_sol["sequence"]
    sequence = convertsequence(sequence, env)
    counterdict = Dict(m => 1 for m in env.problem.machines)

    ExpSequence(sequence, counterdict, false, rng, actionspace, exp_value_sol["objective"]["exp_objective"]) # TODO multiple envs...
end

function test(a::Agent{ExpSequence},env,nrseeds)
    evalfixseq(a, env, a.objective, nrseeds, a.rng)
end

function evalfixseq(agent, env, objective, nrsamples, rng)
    metrics = []
    fitness = []
    objective = -objective

    for i in 1:nrsamples
        agent.approximator.counterdict = Dict(m => 1 for m in env.problem.machines)
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = singleinstance(agent, env, objective, pointer, rng)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
    end

    agent.approximator.counterdict = Dict(m => 1 for m in env.problem.machines)
    return sum(fitness), fitness

end

function singleinstance(agent, env, objective, pointer, rng)
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    t = false
    while !t
        action = nextaction(agent, env)
        # println("action:", env.siminfo.mappingjobs[action[1]]," ", env.siminfo.mappingmachines[action[2]]," ", action[3])
        _, rewards, t, _ = step!(env, action, rng)

        time = env.siminfo.time
        # println("------------------------- $time -------------------------")
        #println("nextevent: ", env.siminfo.nextevent)
        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end

function nextaction(agent::Agent{ExpSequence}, env)
    sequence = agent.approximator.sequence
    counterdict = agent.approximator.counterdict
    actionspace = agent.approximator.actionspace
    for (m,o) in env.state.occupation
        # println(env.state.actionmask[:,m])
        # println(env.state.occupation[m])
        if sum(env.state.actionmask[:,m]) > 0 # otherwise its occupied
            if counterdict[m] > length(sequence[m])
                continue # since machine already finished all its jobs 
            else
                job = sequence[m][counterdict[m]]
                #println(env.state.executable[job])
                if env.state.actionmask[job,m] != 1
                    # println("order next op is: ", env.siminfo.pointercurrentop[job])
                    # println("machine: ", m)
                    # println("the machines that are able to process ne next op are: ", env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]])
                    # println("action ", env.siminfo.mappingjobs[job], " ", env.siminfo.mappingmachines[m] , " not allowed yet")
                    continue # since action is not allowed yet
                elseif env.state.executable[job]
                    if o === nothing
                        counterdict[m] += 1
                        return (job,m,"S")
                    else
                        #println("because machine is occupied")
                        return (job,m,"W")
                    end
                else
                    if actionspace == "AIAR"
                        if env.state.setup[m] !== nothing && env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] !== 0
                            return (job,m,"R")
                        else
                            return (job,m,"W")
                        end
                    else
                        return (job,m,"W")
                    end
                end
            end
        end
    end
    # found no action to act on -> select waiting for any possible action
    cart = findfirst(x -> x == 1, env.state.actionmask)
    return (cart[1],cart[2],"W")
    
end

function convertsequence(sequence, env)
    converted = Dict()

    mapmachineindex = Dict(m => i for (i,m) in env.siminfo.mappingmachines)
    mapjobindex = Dict(j => i for (i,j) in env.siminfo.mappingjobs)
    #convert machines to ints
    for (m, jobs) in sequence
        machine = mapmachineindex[m]
        tmpList = []
        for job in jobs
            append!(tmpList, mapjobindex[job])
        end
        converted[machine] = tmpList
    end
    return converted
end