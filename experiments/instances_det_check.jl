using StatsPlots
using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "default_problem"
Env_AIM_ois = createenv(actionspace = "AIM", actionmask_sorting = ["ois"], instanceType = "doerr", instanceName = instance, deterministic=true)
Env_AIA = createenv(actionspace = "AIA", actionmask_sorting = ["none"], instanceType = "doerr", instanceName = instance, deterministic=true)
Env_AIA_ois = createenv(actionspace = "AIA", actionmask_sorting = ["ois"], instanceType = "doerr", instanceName = instance, deterministic=true)
Env_AIA_ois_uncertain = createenv(actionspace = "AIA", actionmask_sorting = ["ois"], instanceType = "doerr", instanceName = instance, deterministic=false)

testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "doerr" ,
                    "instancename" =>instance)]

testenvs_AIM_ois = generateset(testenvs, n_timesamples=100, actionspace = "AIM", actionmask_sorting = ["ois"])
testenvs_AIA = generateset(testenvs, n_timesamples=100, actionspace = "AIA")
testenvs_AIA_ois = generateset(testenvs, n_timesamples=100, actionspace = "AIA", actionmask_sorting = ["ois"])

# Exp. sequence AIA
sequence_AIA = createagent(createExpSequence(instance ,Env_AIA, "AIA", objective="makespan"), "AIA")
results_seq_AIA = testagent(sequence_AIA, testenvs_AIA)
println("Exp. sequence AIA: ", results_seq_AIA)

Agent_GP_AIA = createagent(createGP([Env_AIA_ois], stationary_samples = true, samples_generation = 1, 
                                            markerdensity = true), "AIA", weight_makespan = 1.0, weight_tardiness = 0.0)
results_GP_AIA = trainagent!(Agent_GP_AIA, generations = 100, finaleval = true, testenvs = testenvs_AIA_ois, TBlog = true)

# update env
setenvs!(Agent_GP_AIA, [Env_AIA_ois_uncertain])
# increase samples
results_GP_AIA_uncertain = trainagent!(Agent_GP_AIA, generations = 30, finaleval = true, testenvs = testenvs_AIA_ois, TBlog = true)

