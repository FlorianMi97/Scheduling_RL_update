using StatsPlots
using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "dpg_def_e70"
Env_AIA = createenv(actionspace = "AIA", instanceType = "doerr", instanceName = instance)
# Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------

testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "doerr" ,
                    "instancename" => instance)]

testenvs_AIA = generateset(testenvs, n_timesamples=100, actionspace = "AIA")

sequence_AIA_mean = createagent(createExpSequence(instance, Env_AIA, actionspace="AIA", exp_measure="mean", objective="makespan"), "AIA", weight_makespan=1.0, weight_tardiness=0.0)
results_seq_AIA_mean = testagent(sequence_AIA_mean, testenvs_AIA)
println("Exp. sequence AIA mean: ", results_seq_AIA_mean)

sequence_AIA_median = createagent(createExpSequence(instance, Env_AIA, actionspace="AIA", exp_measure="median", objective="makespan"), "AIA", weight_makespan=1.0, weight_tardiness=0.0)
results_seq_AIA_median = testagent(sequence_AIA_median, testenvs_AIA)
println("Exp. sequence AIA median: ", results_seq_AIA_median)

# boxplot and show mean
boxplot([results_seq_AIA_mean[3], results_seq_AIA_median[3]],
        label = ["Exp Seq mean" "Exp Seq median"],
        title = string("Boxplot of ", instance,  " makespan"),
        ylabel = "makespan of samples",
        xlabel = "model")

scatter!([1], [results_seq_AIA_mean[1]], label="mean Exp Seq mean", color="red")
scatter!([2], [results_seq_AIA_median[1]], label="mean Exp Seq median", color="blue")

        
