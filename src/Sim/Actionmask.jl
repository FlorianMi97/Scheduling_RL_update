"""initial action mask based on problem"""
function defineactionmask(problem::Problem, am::Array{String})
    maxj = length(problem.orders)
    maxm = length(problem.machines)
    # generate matrix of size jobs/orders x machines
    actionmask = fill(0, (maxj, maxm))
    
    # change entries based on masking am
    # TODO add all others
    # dds = due date sorting of jobs of a same type comepting for a set of machines
    if "dds" ∈ am || "ois" ∈ am
        sortingby = "dds" ∈ am ? "d" : "o"
        smallestcriteria = Dict(type => Dict(machine => Inf for machine ∈ keys(problem.mapindexmachine)) for type ∈ keys(problem.mapindexprods))
        for j ∈ 1:maxj
            o = problem.orders[j]
            for m ∈ o.eligmachines[o.ops[1]]
                if sortingby == "o"
                    if parse(Int64, o.id[5:end]) < smallestcriteria[o.type][m]
                        smallestcriteria[o.type][m] = parse(Int64, o.id[5:end])
                    end
                else
                    if o.duedate < smallestcriteria[o.type][m]
                        smallestcriteria[o.type][m] = o.duedate
                    end
                end
            end
        end

        for j ∈ 1:maxj
            o = problem.orders[j]
            for m ∈ o.eligmachines[o.ops[1]]
                if sortingby == "o"
                    if parse(Int64, o.id[5:end]) == smallestcriteria[o.type][m]
                        actionmask[j,m] = 1
                    end
                else
                    if o.duedate == smallestcriteria[o.type][m]
                        actionmask[j,m] = 1
                    end
                end
            end
        end
    else
        for j ∈ 1:maxj
            o = problem.orders[j]
            for m ∈ o.eligmachines[o.ops[1]]
                actionmask[j,m] = 1
            end
        end 
    end
    return actionmask
end

"""
for updating action mask if an action is taken
"""
function updatedactionmask!(env::Env, action::Tuple{Int,Int,String})
    # TODO different logic when "due date sorting of jobs of a same type competing for a set of machines"

    job = action[1]
    machine = action[2]
    env.state.actionmask[job,:] .= 0 # remove job from actionmask
    env.state.actionmask[:,machine] .= 0  # remove machine from actionmask

    sorting = "dds" ∈ env.actionmasks || "ois" ∈ env.actionmasks
    if sorting
        sortingtype = "dds" ∈ env.actionmasks ? "d" : "o"
    end

    if action[3] == "S" && env.actionspace != "AIM"
        o = env.problem.orders[job]
        if env.siminfo.pointercurrentop[job] != 0
            for m ∈ o.eligmachines[o.ops[env.siminfo.pointercurrentop[job]]] # this includes jobs scheduled on machine before for AIA
                if sum(env.state.actionmask[:,m]) > 0
                    if (env.state.executable[job] || (env.state.occupation[m] === nothing && hasexecutablejob(env, m)))
                        if !sorting || sortingtype == "d" && env.siminfo.edd_type_machine[o.type][m]["waiting"] == o.duedate || sortingtype == "o" && env.siminfo.id_type_machine[o.type][m]["waiting"] == parse(Int64, o.id[5:end])
                            env.state.actionmask[job,m] = 1
                        end
                    elseif env.actionspace == "AIAR" && env.state.occupation[m] === nothing
                        if !sorting || sortingtype == "d" && env.siminfo.edd_type_machine[o.type][m]["waiting"] == o.duedate || sortingtype == "o" && env.siminfo.id_type_machine[o.type][m]["waiting"] == parse(Int64, o.id[5:end])
                            env.state.actionmask[job,m] = 1
                        end
                    end
                end
            end
        end
    end
end

"""
for updating action mask if time elapsed
"""
function updatedactionmask!(env::Env)

    sorting = "dds" ∈ env.actionmasks || "ois" ∈ env.actionmasks
    if sorting
        sortingtype = "dds" ∈ env.actionmasks ? "d" : "o"
    end
    
    maxj = length(env.problem.orders)

    for j ∈ 1:maxj
        if env.siminfo.pointercurrentop[j] != 0
            o = env.problem.orders[j]
            machines = o.eligmachines[o.ops[env.siminfo.pointercurrentop[j]]]
            if readymachine(machines, env.state.occupation) || (env.actionspace != "AIM")  # waitign actions ∈ not AIM spaces!
                for m ∈ machines # this includes jobs scheduled on machine before for AIA
                    if env.state.occupation[m] === nothing && env.state.executable[j] == 1 # scheduling
                        if !sorting || sortingtype == "d" && env.siminfo.edd_type_machine[o.type][m]["scheduling"] == o.duedate || sortingtype == "o" && env.siminfo.id_type_machine[o.type][m]["scheduling"] == parse(Int64, o.id[5:end])
                            env.state.actionmask[j,m] = 1
                        end
                    elseif env.actionspace != "AIM" && (env.state.executable[j] || (env.state.occupation[m] === nothing && hasexecutablejob(env, m))) # waiting only if there is a job executable on the machine 
                        if !sorting || sortingtype == "d" && env.siminfo.edd_type_machine[o.type][m]["waiting"] == o.duedate || sortingtype == "o" && env.siminfo.id_type_machine[o.type][m]["waiting"] == parse(Int64, o.id[5:end])
                            env.state.actionmask[j,m] = 1
                        end
                    elseif env.actionspace == "AIAR" && env.state.occupation[m] === nothing  # doing a risk setup
                        if !sorting || sortingtype == "d" && env.siminfo.edd_type_machine[o.type][m]["waiting"] == o.duedate || sortingtype == "o" && env.siminfo.id_type_machine[o.type][m]["waiting"] == parse(Int64, o.id[5:end])
                            env.state.actionmask[j,m] = 1
                        end
                    end
                end
            end
        end
    end
end

function updateactionmask_edd_change(env::Env, action::Tuple{Int,Int,String}, changesS, changesW)
    job = action[1]
    o = env.problem.orders[job]
    if !isempty(changesS)
        for m ∈ changesS
            if sum(env.state.actionmask[:,m]) > 0
                for j ∈ 1:length(env.problem.orders)
                    oj = env.problem.orders[j]
                    if oj != o
                        if env.siminfo.pointercurrentop[j] != 0
                            if m ∈ oj.eligmachines[oj.ops[env.siminfo.pointercurrentop[j]]]
                                if oj.type == o.type
                                    if oj.duedate == env.siminfo.edd_type_machine[o.type][m]["schedule"]
                                        if env.state.occupation[m] === nothing && env.state.executable[j] == 1 # scheduling
                                            env.state.actionmask[j,m] = 1
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    if env.actionspace != "AIM"
        if !isempty(changesW)
            for m ∈ changesW
                if sum(env.state.actionmask[:,m]) > 0
                    for j ∈ 1:length(env.problem.orders)
                        oj = env.problem.orders[j]
                        if oj != o
                            if env.siminfo.pointercurrentop[j] != 0
                                if m ∈ oj.eligmachines[oj.ops[env.siminfo.pointercurrentop[j]]]
                                    if oj.type == o.type
                                        if oj.duedate == env.siminfo.edd_type_machine[o.type][m]["waiting"]
                                            if (env.state.executable[j] || (env.state.occupation[m] === nothing && hasexecutablejob(env, m))) # waiting only if there is a job executable on the machine
                                                env.state.actionmask[j,m] = 1
                                            elseif env.actionspace == "AIAR" && env.state.occupation[m] === nothing  # doing a risk setup
                                                env.state.actionmask[j,m] = 1
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function updateactionmask_id_change(env::Env, action::Tuple{Int,Int,String}, changesS, changesW)
    job = action[1]
    o = env.problem.orders[job]
    if !isempty(changesS)
        for m ∈ changesS
            if sum(env.state.actionmask[:,m]) > 0
                for j ∈ 1:length(env.problem.orders)
                    oj = env.problem.orders[j]
                    if oj != o
                        if env.siminfo.pointercurrentop[j] != 0
                            if m ∈ oj.eligmachines[oj.ops[env.siminfo.pointercurrentop[j]]]
                                if oj.type == o.type
                                    if parse(Int64, o.id[5:end]) == env.siminfo.id_type_machine[o.type][m]["schedule"]
                                        if env.state.occupation[m] === nothing && env.state.executable[j] == 1 # scheduling
                                            env.state.actionmask[j,m] = 1
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    if env.actionspace != "AIM"
        if !isempty(changesW)
            for m ∈ changesW
                if sum(env.state.actionmask[:,m]) > 0
                    for j ∈ 1:length(env.problem.orders)
                        oj = env.problem.orders[j]
                        if oj != o
                            if env.siminfo.pointercurrentop[j] != 0
                                if m ∈ oj.eligmachines[oj.ops[env.siminfo.pointercurrentop[j]]]
                                    if oj.type == o.type
                                        if parse(Int64, o.id[5:end]) == env.siminfo.id_type_machine[o.type][m]["waiting"]
                                            if (env.state.executable[j] || (env.state.occupation[m] === nothing && hasexecutablejob(env, m))) # waiting only if there is a job executable on the machine
                                                env.state.actionmask[j,m] = 1
                                            elseif env.actionspace == "AIAR" && env.state.occupation[m] === nothing  # doing a risk setup
                                                env.state.actionmask[j,m] = 1
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

"""check if at least on machine is not occupied"""
function readymachine(machines, occupation)
    for m ∈ machines
        if occupation[m] === nothing
            return true
        end
    end
    return false
end

function hasexecutablejob(env, machine)
    for (j,e) ∈ env.state.executable
        if e == 1 
            if machine ∈ env.problem.orders[j].eligmachines[env.problem.orders[j].ops[env.siminfo.pointercurrentop[j]]]
                return true
            end
        end
    end
    return false

end