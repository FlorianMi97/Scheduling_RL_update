using StatsPlots
using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "default_problem_07"
Env_AIA = createenv(actionspace = "AIA", actionmask_sorting = ["ois"], instanceType = "doerr", instanceName = instance)
# Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "doerr" ,
                    "instancename" => instance)]

validationenvs_AIA = generateset(testenvs, n_timesamples=100,fromfile=false, actionspace = "AIA", actionmask_sorting = ["ois"])
testenvs_AIA_ois = generateset(testenvs, n_timesamples=100, actionspace = "AIA", actionmask_sorting = ["ois"])

Agent_GP_AIA_max = createagent(createGP([Env_AIA], stationary_samples=true, samples_generation=20, 
                                            markerdensity=true, antithetic=true, weight_mean=1.0, weight_max=0.0), "AIA", weight_makespan=1.0, weight_tardiness=0.0)
trainagent!(Agent_GP_AIA_max, generations=30, validation=true, validate_every=15, validationenvs=validationenvs_AIA, checkpoint=true, TBlog=true)

results_GP_AIA_max = testagent(Agent_GP_AIA_max, testenvs_AIA_ois)

println("GP_AIA_max: ", results_GP_AIA_max)

# Exp. sequence AIA
Env_AIA_exp = createenv(actionspace="AIA", instanceType="doerr", instanceName=instance)
testenvs_AIA_exp = generateset(testenvs, n_timesamples=100, actionspace="AIA")

sequence_AIA = createagent(createExpSequence(instance ,Env_AIA_exp, "AIA", objective="makespan"), "AIA", weight_makespan=1.0, weight_tardiness=0.0)
results_seq_AIA = testagent(sequence_AIA, testenvs_AIA_exp)
println("Exp. sequence AIA: ", results_seq_AIA)

boxplot([results_seq_AIA[3], results_GP_AIA_max[3]],
        label = ["Exp Seq GP_AIA"],
        title = "Boxplot of default_problem makespan",
        ylabel = "makespan of samples",
        xlabel = "model")
