using StatsPlots
using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "default_problem"
Env_AIA = createenv(actionspace = "AIA", actionmask_sorting = ["dds"], instanceType = "doerr", instanceName = instance)
Env_AIM = createenv(actionspace = "AIM", actionmask_sorting = ["dds"], instanceType = "doerr", instanceName = instance)
# Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "doerr" ,
                    "instancename" =>instance)]

testenvs_AIM = generateset(testenvs, n_timesamples=100, actionspace = "AIM", actionmask_sorting = ["dds"])
testenvs_AIA = generateset(testenvs, n_timesamples=100, actionspace = "AIA")
testenvs_AIA_ois = generateset(testenvs, n_timesamples=100, actionspace = "AIA", actionmask_sorting = ["dds"])
# testenvs_AIAR = generatetestset(testenvs, 100, actionspace = "AIAR")

# Agent_GP_AIA = createagent(createGP([Env_AIA], stationary_samples = true, samples_generation = 30, 
#                                             markerdensity = true, antithetic = true), "AIA", weight_makespan = 1.0, weight_tardiness = 0.0)
# results_GP_AIA = trainagent!(Agent_GP_AIA, generations = 50, evalevery = 25, finaleval = true, testenvs = testenvs_AIA_ois, TBlog = true)

# Agent_GP_AIA_max = createagent(createGP([Env_AIA], stationary_samples = true, samples_generation = 30, 
#                                             markerdensity = true, antithetic = true, weight_mean=0.5, weight_max=0.5), "AIA", weight_makespan = 1.0, weight_tardiness = 0.0)
# results_GP_AIA_max = trainagent!(Agent_GP_AIA_max, generations = 50, evalevery = 25, finaleval = true, testenvs = testenvs_AIA_ois, TBlog = true)

# Agent_GP_AIM_cycling = createagent(createGP([Env_AIM], stationary_samples = true, samples_generation = 30, change_every = 50, steps_in_cycle = 5,
#                                              sample_cycling = true, markerdensity = true, antithetic = true), "AIM", weight_makespan = 1.0, weight_tardiness = 0.0)
# results_GP_AIM = trainagent!(Agent_GP_AIM_cycling, generations = 50, evalevery = 25, finaleval = true, testenvs = testenvs_AIM, TBlog = true)

# Exp. sequence AIA
sequence_AIA = createagent(createExpSequence(instance ,Env_AIA, "AIA", objective="makespan"), "AIA", weight_makespan = 1.0, weight_tardiness = 0.0)
results_seq_AIA = testagent(sequence_AIA, testenvs_AIA)
println("Exp. sequence AIA: ", results_seq_AIA)



# boxplot results
boxplot([results_seq_AIA[3], results_GP_AIA[3],results_GP_AIA_max[3], results_GP_AIM[3]],
        label = ["Exp. Seq" "GP_AIA" "GP_AIA_max" "GP_AIM"],
        title = "Boxplot of default_problem makespan",
        ylabel = "makespan of samples",
        xlabel = "model")