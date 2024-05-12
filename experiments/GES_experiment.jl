using Plots
using StatsPlots
using Flux
using ParameterSchedulers
using Serialization
using SchedulingRL
using ProgressMeter
#--------------------------------------------------------------------------------------------------------------
# Features implemented
# JOB FEATURES
        # "DD"  :  job due date -> never updated use ones.
        # "RO"  :  remaining operations of job 
        # "RW"  :  remaining work of job
        # "ON"  :  average operating time of next operation of job
        # "JS"  :  job slack time (DD - RW - CT)
        # "RF"  :  routing flexibility of remaining operations of job

# MACHINES FEATURES
        # "EW"  :  expected future workload of a machine (proportionally if multiple possible machines)
        # "CW"  :  current workload of a machine (proportionally if multiple possible machines)
        # "JWM" :  Future jobs waiting for machine (proportionally if multiple possible machines)
        # "CJW" :  current jobs waiting (proportionally if multiple possible machines)

# JOB-MACHINES FEATURES
        # "TT"  :  total time of job machine pair (including waiting idle setup processing)
        # "PT"  :  processing time of job machine pair
        # "ST"  :  setup time of job machine pair
        # "NI"  :  required idle time of a machine
        # "NW"  :  needed waiting time of a job
        # "SLA" :  binary: 1 if setupless alternative is available when setup needs to be done, 0 otherwise
        # "BSA" :  binary: 1 if better setup alternative is available, 0 otherwise
        # "DBA" :  returns possitive difference between setup time of current setup and best alternative

# GENERAL
        # "CT"  :  current time

All_Features= ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM","CW", "CJW", "CT","TT", "BSA", "DBA", "RF"]

#objective the agent is trained for,"
#    "[makespan, totalFlowTime, tardiness, discretetardiness, numberTardyJobs, numSetups]" 

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------

#=
instance = "useCase_2_stages"
Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
# Env_AIA = createenv(as = "AIA",instanceType = "usecase", instanceName = instance)
# Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "usecase" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
# testenvs_AIA = generatetestset(testenvs, 100, actionspace = "AIA")
# testenvs_AIAR = generatetestset(testenvs, 100, actionspace = "AIAR")
=#

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#TESTS FOR SOLO GES RUN 
#--------------------------------------------------------------------------------------------------------------
# Tardiness

#=

# SIMPLE RULE DD
println("\n", "TESTED RULE: DD", "\n")
SIMPLE_AIM = createagent(createRule(Priorityfunction("DD")),"AIM")
result_rule_AIM = testagent(SIMPLE_AIM, testenvs_AIM)
println("Result of Rule without specifying weights: ", result_rule_AIM[1], "\n")

priofunction = Priorityfunction("DD")
ges= createGES("global",[Env_AIM], priofunction)
ges.optimizer = "ADAM" # default is "GradientDecent"
Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
println("GES weights initialized: ", Agent_GES_AIM.approximator.bestweights, "\n")
results_GES_AIM = trainagent!(Agent_GES_AIM, generations = 1, evalevery = 1, finaleval = true, showinfo = true, testenvs = testenvs_AIM, termination_crinteria=100)
#println("GES weights after: ", Agent_GES_AIM.approximator.bestweights)
println("GES Result: ", results_GES_AIM[1])

=#

#=

 # SIMPLE RULE RW + EW + TT
 println("\n", "TESTED RULE: PT + DD + RO + RW + ON + JS + ST + NI + NW + SLA + EW + JWM + CT + TT + BSA + DBA + RF", "\n")
 SIMPLE_AIM = createagent(createRule(Priorityfunction("PT + DD + RO + RW + ON + JS + ST + NI + NW + SLA + EW + JWM + CT + TT + BSA + DBA + RF")),"AIM")
 result_rule_AIM = testagent(SIMPLE_AIM, testenvs_AIM)
 #println("Result of Rule without adjusting weights: ", result_rule_AIM[1], "\n")
 
 priofunction = Priorityfunction("PT + DD + RO + RW + ON + JS + ST + NI + NW + SLA + EW + JWM + CT + TT + BSA + DBA + RF" )
 ges= createGES("global",[Env_AIM], priofunction)
 ges.optimizer = "ADAM" # default is "GradientDecent"
 ges.stationary_samples = false
 Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
 #println("GES weights initialized: ", Agent_GES_AIM.approximator.bestweights, "\n")


 results_GES_AIM = trainagent!(Agent_GES_AIM, generations = 50, evalevery = 1, finaleval = true, showinfo = false, showprogressbar = false, TBlog = false, testenvs = testenvs_AIM)
 println("GES Result: ", results_GES_AIM[1])

  # boxplot results
 #graph_makespan = boxplot([result_rule_AIM[10], results_GES_AIM[10]],
 #        label = ["vorher" "nachher"],
 #        title = "Gaps to optimal makespan",
 #        ylabel = "gap",
 #        xlabel = "model")

# tensorboard --logdir="C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/logs/useCase_2_stages"

=#

#--------------------------------------------------------------------------------------------------------------

#=

instance = "useCase_2_stages"
Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "usecase" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")

num_runs = 1
num_generations = 250
features = ["PT + DD + RW + JS + EW + TT + BSA"]
priofunction = Priorityfunction("PT + DD + RW + JS + EW + TT + BSA")
num_weights = numberweights(priofunction)
weights_log = Array{Float32, 2}(undef, num_runs, num_weights)
global Pre_GES_evaluation_result = 0.0
simulation_results = [] #final result of each run
all_rewards = [] #all rewards of all runs over the generations 
calculation_time = [] #times for each generation in each run

for run in 1:num_runs

        println("Run Number: ", run, "\n")
        #Reset the environments
        for env in testenvs_AIM
                resetenv!(env)
        end

        ges= createGES("global",[Env_AIM], priofunction)
        ges.optimizer = "ADAM" 
        ges.stationary_samples = true
        Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
        if run == 1
                global Pre_GES_evaluation_result = testagent(Agent_GES_AIM, testenvs_AIM)[1]
        end
        results_GES_AIM = trainagent!(Agent_GES_AIM, generations = num_generations, evalevery = 1, finaleval = true, showinfo = false, showprogressbar = true, TBlog = false, testenvs = testenvs_AIM)
        #println("GES Result in Simulation run: ", run, " ", results_GES_AIM[1])
        #println("Final weights after Simulation run: ", run, "\n", Agent_GES_AIM.approximator.bestweights[1], "\n")
        push!(simulation_results, results_GES_AIM[1])
        push!(all_rewards, copy(Agent_GES_AIM.logger["reward"]))
        for weight in 1:num_weights
                weights_log[run, weight] = Agent_GES_AIM.approximator.bestweights[1][weight]
        end
        push!(calculation_time, copy(Agent_GES_AIM.logger["elapsed_time_per_gen"]))
        println(results_GES_AIM[1])
end
#inner_array = calculation_time[1]
#calculation_time_seconds = [millis.value / 1000 for millis in inner_array]
#plot(calculation_time_seconds, label="Calculation Time", title="Calculation Time per Generation",
#     ylabel="Time in seconds", xlabel="Generation")
#println("sum calculation time: ", sum(calculation_time_seconds))
save_weights_boxplot(weights_log, features, "C:\\Users\\floppy\\Desktop\\Boxplots\\weights_boxplot.png")
save_results_diagram(simulation_results, Pre_GES_evaluation_result, "C:\\Users\\floppy\\Desktop\\Boxplots\\results_boxplot.png")
save_objective_evolution(all_rewards, "C:\\Users\\floppy\\Desktop\\Boxplots\\objective_evolution.png")



=#


#println(all_rewards)
#=
#save the data
serialize("weights_log.jls", weights_log)
serialize("simulation_results.jls", simulation_results)
serialize("all_rewards.jls", all_rewards)
# Load the data
weights_log = deserialize("weights_log.jls")
simulation_results = deserialize("simulation_results.jls")
all_rewards = deserialize("all_rewards.jls")

num_runs = 20
invalid_runs_indices = [13,17]
valid_runs_indices = setdiff(1:num_runs, invalid_runs_indices)
weights_log_filtered = weights_log[valid_runs_indices, :]
simulation_results_filtered = [simulation_results[i] for i in valid_runs_indices]
all_rewards_filtered = [all_rewards[i] for i in valid_runs_indices]


save_weights_boxplot(weights_log_filtered, features, "C:\\Users\\floppy\\Desktop\\Boxplots\\weights_boxplot.png")
save_results_diagram(simulation_results_filtered, Pre_GES_evaluation_result, "C:\\Users\\floppy\\Desktop\\Boxplots\\results_boxplot.png")
save_objective_evolution(all_rewards_filtered, "C:\\Users\\floppy\\Desktop\\Boxplots\\objective_evolution.png")
=#
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
#TESTS FOR GP INTO GES RUNS 



# Total Tardiness

instance = "useCase_2_stages"
#instance = "default_problem"
Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
#Env_AIM = createenv(as = "AIM", instanceName = "lhmhlh-0_ii_Expo5")
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "usecase" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]
#testenvs = [Dict("layout" => "Flow_shops" ,
#                    "instancetype" => "benchmark" ,
#                    "instancesettings" => "base_setting",
#                    "datatype" => "data_ii",
#                    "instancename" =>"lhmhlh-0_ii_Expo5")]

testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
#println("Env type: ", testenvs_AIM[1].type)
#--------------------------------------------------------------------------------------------------------------
# SIMPLE RULE: EDD
#EDD_AIM = createagent(createRule(Priorityfunction("DD")),"AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
#result_rule_AIM = testagent(EDD_AIM,testenvs_AIM)
#println("EDD AIM: ", result_rule_AIM[6])

#--------------------------------------------------------------------------------------------------------------
#GP

#GP_complex = createGP([Env_AIM],simplerules = false)
#AgentGP_complex = createagent(GP_complex, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
#resultsGP_complex = trainagent!(AgentGP_complex,generations = 50, evalevery = 0, finaleval = true, testenvs = testenvs_AIM)

#GP_simple_benachmark = createGP([Env_AIM],simplerules = true)
#AgentGP_simple_benachmark = createagent(GP_simple_benachmark, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
#resultsGP_simple_benchmark = trainagent!(AgentGP_simple_benachmark,generations = 50, evalevery = 0, finaleval = true, testenvs = testenvs_AIM)


complex_benchmark_experiment = deserialize("complex_GP_example_runs_results.jls")
simple_benchmark_experiment = deserialize("simple_benchmark_GP_example_runs_results.jls")
simple_experiment = deserialize("simple_GP_example_runs_results.jls")

complex_benchmark_minimal_results = [complex_benchmark_experiment[run][2][1] for run in eachindex(complex_benchmark_experiment)]
simple_benchmark_minimal_results = [simple_benchmark_experiment[run][2][1] for run in eachindex(simple_benchmark_experiment)]
simple_experiment_minimal_results = [simple_experiment[run][2][1] for run in eachindex(simple_experiment)]

boxplot([complex_benchmark_minimal_results, simple_benchmark_minimal_results, simple_experiment_minimal_results], label=["Complex Benchmark", "Simple Benchmark", "Simple Experiment"], title="Minimal Rewards per Run",
     ylabel="Minimal Reward over runs", xlabel="Type")     
#AgentGp_Complex_Priofunction= simplify_expression((get_expression(simple_GP_example_runs_results[1][1].model[1][1][1], simple_GP_example_runs_results[1][1].approximator)), simple_GP_example_runs_results[1][1].approximator)
#drawTree(Priofunction, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\simple_GP_example_runs_results.png")

#AgentGP_simple_benachmark_Priofunction = simplify_expression((get_expression(AgentGP_simple_benachmark.model[1][1][1], AgentGP_simple_benachmark.approximator)), AgentGP_simple_benachmark.approximator)
#drawTree(AgentGP_simple_benachmark_Priofunction, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Erster_GP_Test\\AgentGP_simple_benachmark_Priofunction.png")
#AgentGP_simple_benachmark_Priofunction = simplify_expression((get_expression(AgentGP_simple_benachmark.model[1][1][1], AgentGP_simple_benachmark.approximator)), AgentGP_simple_benachmark.approximator)
#drawTree(AgentGP_simple_benachmark_Priofunction, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Erster_GP_Test\\AgentGP_simple_benachmark_Priofunction.png")

#GP_complex_best_priority_function = get_expression(AgentGP_complex.model[1][1][1], AgentGP_complex.approximator)
#drawTree(GP_complex_best_priority_function, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\complexTreePlot.png")
#GP_simple_benachmark_best_priority_function = simplify_expression(get_expression(AgentGP_simple_benachmark.model[1][1][1], AgentGP_simple_benachmark.approximator), AgentGP_simple_benachmark.approximator)

#latexified = latexify_dispatching_rules(GP_complex_best_priority_function)
#println(latexify(GP_complex_best_priority_function))
#println(GP_simple_benachmark_best_priority_function)
#println(latexify(GP_simple_benachmark_best_priority_function))

#resultsGP_complex = testagent(AgentGP_complex, testenvs_AIM)
#resultsGP_simple_benchmark = testagent(AgentGP_simple_benachmark, testenvs_AIM)
#resultsGP_simple = testagent(AgentGP_simple, testenvs_AIM)

#println("--------------------------------------------------------------------------------------------------------------")
#println("GP simple Result: ", resultsGP_simple[1])


#plot(complex_benchmark_experiment[i][2][1] for i in complex_benchmark_experiment, label="Complex Benchmark", title="Minimal Rewards per Run",
#     ylabel="Reward", xlabel="Run")

#save_results_diagram(simple_GP_example_runs_results, AgentGP_complex[1], AgentGP_simple_benachmark[1], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\simple_GP_example_runs_results.png")

#=

#--------------------------------------------------------------------------------------------------------------

# # GES

#------------------------------------------
#params for global Rules

#granularity = "global"
#AgentGP.model[1][1][1] #GP best rule
#priorules = Priorityfunction("TT + EW + RW + JS + DD + PT + ST")

#------------------------------------------
#params for stage Rules

#granularity = "stage"
#AgentGP.model[1][1] #GP best rules

#priorule1 = Priorityfunction("TT + EW + RW + JS + DD")
#priorule2 = Priorityfunction("TT + EW + RW + JS + DD + ST")
#priorules = Vector([priorule1, priorule2])
#------------------------------------------
# granulariry = "resource"

#------------------------------------------
#Get GP output to initialize GES with a simplified (linear) version of it
GP_best_priority_function = get_expression(AgentGP_simple.model[1][1][1], AgentGP_simple.approximator)
#test_of_GP_best_rule = testfitness([GP_best_priority_function], testenvs_AIM[1], AgentGP_simple.objective, 30, AgentGP_simple.rng)[1:3]
#println("Test Result of GP best rule: ", test_of_GP_best_rule[1], "\n")
#println("GP prio function expression: ", GP_best_priority_function)
#drawTree(GP_best_priority_function, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\TreePlot.png")


Simpliefied_best_Rule = simplify_expression(GP_best_priority_function, AgentGP_simple.approximator)
#test_of_simplified_rule = testfitness([Simpliefied_best_Rule], testenvs_AIM[1], AgentGP_simple.objective, 30, AgentGP_simple.rng)[1:3]
#println("Test Result of simplified rule: ", test_of_simplified_rule[1], "\n")
#println("Simplified prio function expression: ", Simpliefied_best_Rule)
drawTree(Simpliefied_best_Rule, 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\SimplyfiedTreePlot.png")

#Create a priorityfunction from the simplified best rule (priorules::expr, priofunction_strig::String)
featureToPrimitives, numvalue, FeatureCounter = analyze_tree(Simpliefied_best_Rule)
#println("Feaature_to_Primitives: ", Analyze_Tree[1], "\n")
Initial_weights, Initial_weights_normalized, weightssigns = Covert_weights_from_GP(featureToPrimitives, FeatureCounter)
#println("Initial weights: ", Initial_weights)
#println("Initial weights normalized: ", Initial_weights_normalized)
#println("Initial weight signs: ", weightssigns)
priorules, priofunction_string = PriorityFunctionFromTree(Simpliefied_best_Rule, true)
#println("All Features: ", All_Features)
#println("Priority Function mask: ", priorules.nn_mask, "\n")
#println("priofunction_string that is not normalized : ", priofunction_string, "\n")



#=


# create GES
ges = createGES("global", [Env_AIM], priorules)

#------------------------------------------
# initial parameters and setup GES

Analyze_simplified_Tree = analyze_tree(Simpliefied_best_Rule)
Initial_best_weights = Covert_weights_from_GP(Analyze_simplified_Tree[1], Analyze_simplified_Tree[3])
s = Vector{Vector{Float32}}([fill(log(3), numberweights(priorules))]) # source-paper sets all sigma to log(3)
ges.optimizer = "ADAM"
#ges.stationary_samples = false
ges.bestweights = [Initial_best_weights]
ges.sigma = s

#setinitalparameters(ges, initialweights = Initial_best_weights, sigma = s)
#println("Mask of priorityfunction: ", priorules.nn_mask, "\n")
#println("Constant of priorityfunction: ", priorules.constant, "\n")
#println("GES initial best weights: ", ges.bestweights, "\n")
#println("GES initial sigma: ", ges.sigma, "\n")


#------------------------------------------
# create agent and learning process 

Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
results_GES_AIM = trainagent!(Agent_GES_AIM, generations = 1, evalevery = 1, finaleval = true, showinfo = false, showprogressbar = false, TBlog = false, testenvs = testenvs_AIM)
p1 = plot(Agent_GES_AIM.logger["reward"], label="GES", title="Rewards per Generation",
     ylabel="Reward", xlabel="Generation")
savefig(p1, "C:\\Users\\floppy\\Desktop\\Experimente\\Erster_GP_Test\\GES_objective_value.png")
#println("GES AIM Result: " , results_GES_AIM[1])

=#


num_runs = 1
num_generations = 1000
num_weights = numberweights(priorules)
weights_log = Array{Float32, 2}(undef, num_runs, num_weights)
simulation_results = [] #final result of each run
all_rewards = [] #all rewards of all runs over the generations 
gaps = [] #all gaps of all runs 
#calculation_time = [] #times for each generation in each run

#println("Number of Threads used: ",Threads.nthreads())

# Create a progress bar
#progress = Progress(num_runs, 1, "Learning in GES runs...")

for run in 1:num_runs
    # Reset the environments
    for env in testenvs_AIM
        resetenv!(env)
    end

    ges = createGES("global", [Env_AIM], priorules)
    Analyze_simplified_Tree = analyze_tree(Simpliefied_best_Rule)
    Initial_best_weights, Initial_best_weights_normalized, Initial_weight_signs= Covert_weights_from_GP(Analyze_simplified_Tree[1], Analyze_simplified_Tree[3])
    s = Vector{Vector{Float32}}([fill(log(3), num_weights)]) # source-paper sets all sigma to log(3)
    #ges.decay_SGD = true
    ges.optimizer = "ADAM"
    #ges.stationary_samples = false
    ges.bestweights = [Initial_best_weights_normalized]
    ges.weights_with_signs = [Initial_best_weights]
    #println("ges intial weights: ", ges.bestweights)
    #println("ges intial weights with signs: ", ges.weights_with_signs)
    ges.sigma = s

    Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
    results_GES_AIM = trainagent!(Agent_GES_AIM, generations = num_generations, evalevery = 1, finaleval = true, showinfo = false, showprogressbar = true, TBlog = false, testenvs = testenvs_AIM, termination_criteria=1000)
    push!(simulation_results, results_GES_AIM[1])
    push!(gaps, results_GES_AIM[3])
    push!(all_rewards, copy(Agent_GES_AIM.logger["reward"]))
    p = plot(Agent_GES_AIM.logger["reward"], label="Run $run", title="\n Evolution of Objective Value Over Generations",
        ylabel="Average Objective Value", xlabel="Generation", legend=:topright, fontfamily="Computer Modern", fontsize=6 )#xticks=1:25:num_generations
    savefig(p, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\objective evolution for each run\\objective evolution run $run.png")
    #savefig(p, "C:\\Users\\ge24kom\\Experimente\\GP_5_into_20GESRuns\\single runs\\objective evolution run $run.png") #remote
    for weight in 1:num_weights
        weights_log[run, weight] = Agent_GES_AIM.approximator.weights_with_signs[1][weight]
    end
    #ProgressMeter.next!(progress) 
    #push!(calculation_time, copy(Agent_GES_AIM.logger["elapsed_time_per_gen"]))
end



#inner_array = calculation_time[1]
#calculation_time_seconds = [millis.value / 1000 for millis in inner_array]
#plot(calculation_time_seconds, label="Calculation Time", title="Calculation Time per Generation",
#     ylabel="Time in seconds", xlabel="Generation")
#println("sum calculation time: ", sum(calculation_time_seconds))

#boxplot results

save_weights_boxplot(weights_log, priorules, priofunction_string, Initial_weights, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\weights_boxplot.png")
save_results_diagram(simulation_results, resultsGP_simple[1], resultsGP_simple_benchmark[1] , "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\results_boxplot.png")
save_objective_evolution(all_rewards, resultsGP_simple[1], "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\objective_evolution.png")

#save_weights_boxplot(weights_log, priorules, priofunction_string, Initial_weights, "C:\\Users\\ge24kom\\Experimente\\GP_5_into_20GESRuns\\weights_boxplot.png")
#save_results_diagram(simulation_results, resultsGP_simple[1], "C:\\Users\\ge24kom\\Experimente\\GP_5_into_20GESRuns\\results_boxplot.png")
#save_objective_evolution(all_rewards, "C:\\Users\\ge24kom\\Experimente\\GP_5_into_20GESRuns\\objective_evolution.png")

#b2 = boxplot([resultsGP_complex[3], resultsGP_simple_benchmark[3], resultsGP_simple[3], gaps[argmin(simulation_results)]],
#        label = ["GP benchmark" "GP simple benchmark" "GP simple" "best GES"], title = "\n Boxplot of gaps", ylabel = "gap", xlabel = "model", fontfamily="Computer Modern", fontsize=6)

b2 = boxplot([resultsGP_simple[10], gaps[argmin(simulation_results)]],
        label = ["GP simple" "best GES"], title = "\n Boxplot of gaps", ylabel = "gap", xlabel = "model", fontfamily="Computer Modern", fontsize=6)

savefig(b2, "C:\\Users\\floppy\\Desktop\\Experimente\\GP_5_into_20GESRuns\\Boxplot_of_gaps.png")
#savefig(b2, "C:\\Users\\ge24kom\\Experimente\\GP_5_into_20GESRuns\\Boxplot_of_gaps.png")




=#





#--------------------------------------------------------------------------------------------------------------
# TEST AND TRY

# Test if the best priority function is evaluated the same way as the simplified one --> yes in this case
#=
GP_best_priority_function = get_expression([SchedulingRL.add, SchedulingRL.add, SchedulingRL.add, :CT, nothing, nothing, nothing, nothing, nothing, nothing, :ST, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, :JS, nothing, nothing, nothing, nothing, nothing, nothing, :RF, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, SchedulingRL.add, 0.02, nothing, nothing, nothing, nothing, nothing, nothing, :JS, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.sub, 0.413, nothing, nothing, nothing, nothing, nothing, nothing, :RF, nothing, nothing, nothing, nothing, nothing, nothing],AgentGP.approximator)
println("GP best priority function: ", GP_best_priority_function, "\n")
test_of_best_rule = testfitness([GP_best_priority_function], testenvs_AIM[1], AgentGP.objective, 30, AgentGP.rng)[1:3]
println("Test Result Priority Rule: ", test_of_simplified_rule[1], "\n")

Simpliefied_best_Rule = simplify_expression(GP_best_priority_function,AgentGP.approximator)
println("Simpliefied_best_Rule: \n", Simpliefied_best_Rule , "\n")
#drawTree(Simpliefied_best_Rule,1)
test_of_simplified_rule = testfitness([Simpliefied_best_Rule], testenvs_AIM[1], AgentGP.objective, 30, AgentGP.rng)[1:3]
println("Test Result of simplified rule: ", test_of_simplified_rule[1], "\n")
=#

# COMPARISON OF GP AND GES INITIAL RESULTS
#--> Showed differnt results for the same rule 
#--> Debugged: 
# 1) Rounding of the values in the prioritymatrix of the GES solved the problem 

#= complex_benchmark_experiment = deserialize("complex_GP_example_runs_results.jls")
simple_benchmark_experiment = deserialize("simple_benchmark_GP_example_runs_results.jls")
simple_experiment = deserialize("simple_GP_example_runs_results.jls")

complex_benchmark_minimal_results = [complex_benchmark_experiment[run][2][1] for run in eachindex(complex_benchmark_experiment)]
simple_benchmark_minimal_results = [simple_benchmark_experiment[run][2][1] for run in eachindex(simple_benchmark_experiment)]
simple_experiment_minimal_results = [simple_experiment[run][2][1] for run in eachindex(simple_experiment)]

simple_experiments_agents = [simple_experiment[i][1] for i in eachindex(simple_experiment)]
#take all agents and test them and save all the agents perfomances
simple_GP_example_runs_results = [testagent(agent, validationenvs_AIM) for agent in simple_experiments_agents]
simplified_best_rules = [simplify_expression(get_expression(agent.model[1][1][1], agent.approximator), agent.approximator) for agent in simple_experiments_agents]
analyzed_simplfied_best_rules = [analyze_tree(rule) for rule in simplified_best_rules]
inital_weights = [Covert_weights_from_GP(analyzed_simplfied_best_rules[i][1], analyzed_simplfied_best_rules[i][3]) for i in eachindex(analyzed_simplfied_best_rules)]
#For all simplified rules create priorityfunctions 
Priorityfunctions = [PriorityFunctionFromTree(simplified_best_rules[i], true) for i in eachindex(simplified_best_rules)]
#create ges for each agent and initialize with normalized weights
ges = [createGES("global", [Env_AIM], Priorityfunctions[i][1]) for i in eachindex(Priorityfunctions)]
for i in eachindex(ges)
    ges[i].optimizer = "ADAM"
    ges[i].bestweights = [inital_weights[i][2]] #normalized weights
    ges[i].weights_with_signs = [inital_weights[i][1]]
end
#test each ges agent without training 
ges_agents = [createagent(ges[i], "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0]) for i in eachindex(ges)]
results_GES_AIM = [testagent(agent, validationenvs_AIM) for agent in ges_agents]
# plot the results of the teste GP agents versus the tested GES agents
plot([simple_GP_example_runs_results[i][1] for i in eachindex(simple_GP_example_runs_results)], label="Simple GP", title="Comparison of GP and GES Initial Results",
     ylabel="Reward", xlabel="Run")
plot!([results_GES_AIM[i][1] for i in eachindex(results_GES_AIM)], label="GES")
savefig("C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Comparison of GP and GES Initial Results.png")

#draw the tree of the best rule of gp agent
#drawTree(simplified_best_rules[3], 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\simplified_best_rule.png")
#println("Priofunction Expression: ", Meta.show_sexpr(Priorityfunctions[1][1].expression))
create_and_display_rules_table(simple_GP_example_runs_results, results_GES_AIM, Priorityfunctions, inital_weights, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\rules_table.html") =#

#--------------------
# COMPARISON OF GP AND GES Validation: priomatrices


#= include("C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/src/Approximators/GES_GEP/UtilsGES.jl")
include("C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/src/Approximators/GP/Wrapper_GP.jl")
include("C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/src/Approximators/GES_GEP/Wrapper_Prio_rule.jl")

Env_AIM_2 = deepcopy(Env_AIM)
validationenvs_AIM_2 = deepcopy(validationenvs_AIM)

println("----------------------------------------------------")
complex_benchmark_experiment = deserialize("complex_GP_example_runs_results.jls")
simple_benchmark_experiment = deserialize("simple_benchmark_GP_example_runs_results.jls")
simple_experiment = deserialize("simple_GP_example_runs_results.jls")

simple_experiments_agents = [simple_experiment[i][1] for i in eachindex(simple_experiment)]
Individual_rules = [get_expression(simple_experiments_agents[i].model[1][1][1], simple_experiments_agents[i].approximator) for i in eachindex(simple_experiments_agents)]
#println("GP Expression of the Rule: ", "\n", Individual_rules[14])
#drawTree(Individual_rules[14], 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Best_Rule.png")

#action_GP, priomatrix_GP = actionsfromindividuum([Individual_rules[14]], validationenvs_AIM[1]) #rule 1
#println("Action GP: ", action_GP)

simplified_best_rules = [simplify_expression(get_expression(agent.model[1][1][1], agent.approximator), agent.approximator) for agent in simple_experiments_agents]
#drawTree(simplified_best_rules[14], 1, "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\simplified_best_rule.png")
analyzed_simplfied_best_rules = [analyze_tree(rule) for rule in simplified_best_rules]
inital_weights = [Covert_weights_from_GP(analyzed_simplfied_best_rules[i][1], analyzed_simplfied_best_rules[i][3]) for i in eachindex(analyzed_simplfied_best_rules)]
Priorityfunctions = [PriorityFunctionFromTree(simplified_best_rules[i], true) for i in eachindex(simplified_best_rules)]
ges = [createGES("global", [Env_AIM], Priorityfunctions[i][1]) for i in eachindex(Priorityfunctions)]
for i in eachindex(ges)
        ges[i].bestweights = [inital_weights[i][2]] #normalized weights
end
ges_agents = [createagent(ges[i], "AIM", obj=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) for i in eachindex(ges)]

#action_GES, priomatrix_GES = translateaction(ges_agents[14].approximator.priorityfunctions, ges_agents[14].approximator.bestweights, validationenvs_AIM_2[1])
#println("Action GES: ", action_GES)

resetenv!(validationenvs_AIM[1])
setsamplepointer!(validationenvs_AIM[1].siminfo, 1)
resetenv!(validationenvs_AIM_2[1])
setsamplepointer!(validationenvs_AIM_2[1].siminfo, 1)

for state = 1:100
        println("State: ", state)
        action_GP, priomatrix_GP, indices_GP, mininimum_matrix_GP, possible_action_GP = actionsfromindividuum([Individual_rules[14]], validationenvs_AIM[1])
        #df_GP = DataFrame(priomatrix_GP, :auto)
        #Save_as_html_content(df_GP, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GP_state_$state.html", true)
        println("Action GP: ", action_GP)
        action_GES, priomatrix_GES, indices_GES, mininimum_matrix_GES, possible_action_GES = translateaction(ges_agents[14].approximator.priorityfunctions, ges_agents[14].approximator.bestweights, validationenvs_AIM_2[1])
        #df_GES = DataFrame(priomatrix_GES, :auto)
        #Save_as_html_content(df_GES, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GES_state_$state.html", true)
        println("Action GES: ", action_GES)
        println("-----")
        if action_GP != [action_GES] #indices_GP != indices_GES                      
                println("\n", "Actions first differ in State: ", state)
                #println("minimum matrix indices in GP are: ",mininimum_matrix_GP)
                #println("Action GP: ", action_GP, "\n")
                #println("minimum matrix indices in GES are: ",mininimum_matrix_GES)
                #println("Action GES: ", action_GES, "\n")
                #println("Priomatrix GP: ", "\n",  priomatrix_GP, "\n")
                #println("Priomatrix GES: ", "\n", priomatrix_GES, "\n")
                #df_GP = DataFrame(priomatrix_GP, :auto)
                #Save_as_html_content(df_GP, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GP_final.html", true)
                #df_GES = DataFrame(priomatrix_GES, :auto)
                #Save_as_html_content(df_GES, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GES_final.html", true)
                break
        else
                nextstate_GP, rewards_GP, t_GP, info_GP = step!(validationenvs_AIM[1], action_GP[1], simple_experiments_agents[14].approximator.rng)
                nextstate_GES, rewards_GES, t_GES, info_GES = step!(validationenvs_AIM_2[1], action_GES, ges_agents[14].approximator.rng)
        end
        if state == 100
                println("Final state reached")
        end
end =#

#--------------------
#Test for going step by step in an environment and compare the priomatices of GP and GES

#= n_states = 15
possible_actions_GP = []
possible_actions_GES = []
featurevalues_GP = []
featurevalues_GES = []
actions_GP = []
actions_GES = []

resetenv!(validationenvs_AIM[1])
setsamplepointer!(validationenvs_AIM[1].siminfo, 1)
for state = 1:n_states
        action_GP, priomatrix_GP, indices_GP, mininimum_matrix_GP, possible_action_GP = actionsfromindividuum([Individual_rules[14]], validationenvs_AIM[1])
        df_GP = DataFrame(priomatrix_GP, :auto)
        Save_as_html_content(df_GP, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GP_state_$state.html", true)
        
#=         syms = [symbols(ex) for ex in [Individual_rules[14]]]
        featurefunctions = Dict(:PT => getPT, :DD => getDD, :RO => getRO, :RW => getRW, :ON => getON, :JS => getJS,
        :RF => getRF, :CT => getCT, :EW => getEW, :JWM => getJWM, :ST => getST, :NI => getNI, :NW => getNW, :SLA => getSLA,
        :CW => getCW, :CJW => getCJW, :TT => getTT, :BSA => getBSA, :DBA => getDBA)
        vals_GP = calcprio([Individual_rules[14]], validationenvs_AIM[1], mininimum_matrix_GP[1], mininimum_matrix_GP[2], syms, featurefunctions)[2]
        Feature_symbols = []
        deleated_FS = []
        for feature in All_Features
                push!(Feature_symbols, Symbol(strip(feature, '\"')))
        end
        for features in Feature_symbols
                if haskey(vals_GP, features)
                        push!(deleated_FS, features)
                end
        end
        vals_GP = Dict(k => vals_GP[k] for k in deleated_FS)
        push!(featurevalues_GP, vals_GP) =#

        push!(actions_GP, action_GP)
        push!(possible_actions_GP, possible_action_GP)

        nextstate_GP, rewards_GP, t_GP, info_GP = step!(validationenvs_AIM[1], action_GP[1], simple_experiments_agents[14].approximator.rng)
end

resetenv!(validationenvs_AIM[1])
setsamplepointer!(validationenvs_AIM[1].siminfo, 1)
for state = 1:n_states 
        action_GES, priomatrix_GES, indices_GES, mininimum_matrix_GES, possible_action_GES = translateaction(ges_agents[14].approximator.priorityfunctions, ges_agents[14].approximator.bestweights, validationenvs_AIM[1])
        df_GES = DataFrame(priomatrix_GES, :auto)
        Save_as_html_content(df_GES, ["Job", "Machine"], "C:\\Users\\floppy\\Desktop\\Experimente\\GP Tests\\Simple_GP_Tests\\Priomatices\\priomatrix_GES_state_$state.html", true)
        
#=         f = Featurevalues(validationenvs_AIM[1], mininimum_matrix_GES[1], mininimum_matrix_GES[2])
        featurefunctions_GES = Dict(:PT => Float64(f.PT), :DD => Float64(f.DD), :RO => Float64(f.RO), :RW => Float64(f.RW), :ON => Float64(f.ON), :JS => Float64(f.JS),
        :RF => Float64(f.RF), :CT => Float64(f.CT), :EW => Float64(f.EW), :JWM => Float64(f.JWM), :ST => Float64(f.ST), :NI => Float64(f.NI), :NW => Float64(f.NW), :SLA => Float64(f.SLA),
        :CW => Float64(f.CW), :CJW => Float64(f.CJW), :TT => Float64(f.TT), :BSA => Float64(f.BSA), :DBA => Float64(f.DBA))
        Feature_symbols = []
        for feature in All_Features
                push!(Feature_symbols, Symbol(strip(feature, '\"')))
        end
        deleated_FS = []
        index = 1
        for features in Feature_symbols  
                if ges_agents[14].approximator.priorityfunctions[1].nn_mask[index] == 1
                        push!(deleated_FS, features)
                end
                index += 1
        end
        featurefunctions_GES = Dict(k => featurefunctions_GES[k] for k in deleated_FS)
        push!(featurevalues_GES, featurefunctions_GES) =#

        push!(actions_GES, action_GES)
        push!(possible_actions_GES, possible_action_GES)
        
        nextstate_GES, rewards_GES, t_GES, info_GES = step!(validationenvs_AIM[1], action_GES,  ges_agents[14].approximator.rng) #simple_experiments_agents[14].approximator.rng
end
println("-----")
for state = 1:n_states
        println("State: ", state)
        println("GP", "\n")
        println("Possible actions GP: ", possible_actions_GP[state])
        #println("Featurevalues GP: ", "\n", featurevalues_GP[state])
        println("Action GP: ", actions_GP[state])
        println("-----")
        println("GES", "\n")
        println("Possible actions GES: ", possible_actions_GES[state])
        #println("Featurevalues GES: ", "\n", featurevalues_GES[state])
        println("Action GES: ", actions_GES[state])
        println("------------------")
end =#

#--------------------
# COMPARISON OF GP AND GES Validation: Single values
#=
include("C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/src/Approximators/GP/Wrapper_GP.jl")
include("C:/Users/floppy/Desktop/Mastertheis GitLab Repository/scheduling_RL/src/Approximators/GES_GEP/Wrapper_Prio_rule.jl")
pa = [Tuple(x) for x in findall(x -> x == 1, Env_AIM.state.actionmask)] # possible actions
jobs = unique([x[1] for x in pa])
machines = unique([x[2] for x in pa])

#GP
syms = [symbols(ex) for ex in [Individual_rules[14]]]
featurefunctions = Dict(:PT => getPT, :DD => getDD, :RO => getRO, :RW => getRW, :ON => getON, :JS => getJS,
        :RF => getRF, :CT => getCT, :EW => getEW, :JWM => getJWM, :ST => getST, :NI => getNI, :NW => getNW, :SLA => getSLA,
        :CW => getCW, :CJW => getCJW, :TT => getTT, :BSA => getBSA, :DBA => getDBA)

prio_GP, vals = calcprio([Individual_rules[14]], Env_AIM, jobs[52], machines[2], syms, featurefunctions) #rule 14
println("Featurefunctions GP: ","\n", vals)
println("Prio GP: ", prio_GP, "\n")
println("\n")
#GES
w = Weights(ges[14].bestweights[1], ges[14].priorityfunctions[1].nn_mask)
println("Weights: ", w)
f = Featurevalues(Env_AIM, jobs[52], machines[2])
featurefunctions_GES = Dict(:PT => Float64(f.PT), :DD => Float64(f.DD), :RO => Float64(f.RO), :RW => Float64(f.RW), :ON => Float64(f.ON), :JS => Float64(f.JS),
        :RF => Float64(f.RF), :CT => Float64(f.CT), :EW => Float64(f.EW), :JWM => Float64(f.JWM), :ST => Float64(f.ST), :NI => Float64(f.NI), :NW => Float64(f.NW), :SLA => Float64(f.SLA),
        :CW => Float64(f.CW), :CJW => Float64(f.CJW), :TT => Float64(f.TT), :BSA => Float64(f.BSA), :DBA => Float64(f.DBA))

prio_GES = ges[14].priorityfunctions[1].expression(w, f)
println("GES Expression of the Rule: ", Priorityfunctions[14][1].expressionString)
println("with the weights: ", ges[14].bestweights)
println("Featurevalues for GES: ", "\n", featurefunctions_GES)
println("Prio GES: ", prio_GES, "\n")


#------------------------------------------

# Testing of differnt stages and the recieved actionvector in the GP

#=
GP = createGP([Env_AIM],simplerules = true)
AgentGP = createagent(GP, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])

example_tree_array_1 = Any[SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, :CT, :CW, SchedulingRL.sub, 0.378, :PT, SchedulingRL.sub, SchedulingRL.add, :RW, 0.131, SchedulingRL.add, :NI, :PT, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :RO, :CJW, SchedulingRL.add, 0.462, :CT, SchedulingRL.add, SchedulingRL.sub, 0.102, :RW, SchedulingRL.sub, 0.884, 0.112, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :RF, :BSA, SchedulingRL.sub, :ON, 0.699, SchedulingRL.sub, SchedulingRL.add, :DBA, :RO, SchedulingRL.sub, :CJW, :ST, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.sub, :CW, :TT, SchedulingRL.add, :PT, :CW, SchedulingRL.sub, SchedulingRL.sub, :SLA, 0.592, SchedulingRL.add, :RO, :CT]
example_tree_array_2 = Any[SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :JWM, :TT, SchedulingRL.add, :RF, :CW, SchedulingRL.sub, SchedulingRL.sub, :JWM, 0.593, SchedulingRL.add, :SLA, :RW, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.sub, 0.255, :RW, SchedulingRL.add, :NI, :NI, SchedulingRL.sub, SchedulingRL.sub, :RO, :RF, SchedulingRL.sub, :CT, :DBA, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :JWM, :RW, SchedulingRL.add, :NW, :BSA, SchedulingRL.sub, SchedulingRL.add, :ST, 0.907, SchedulingRL.sub, :JWM, :ST, SchedulingRL.add, SchedulingRL.add, SchedulingRL.add, 0.636, :DD, SchedulingRL.sub, :DBA, :DD, SchedulingRL.sub, SchedulingRL.sub, :ST, 0.42, SchedulingRL.sub, 0.272, 0.04]
example_tree_array_3 = Any[SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, :RF, :CJW, SchedulingRL.add, :ST, :ON, SchedulingRL.add, SchedulingRL.add, 0.919, :RF, SchedulingRL.sub, :ST, :EW, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :BSA, :NI, SchedulingRL.add, :RF, :JS, SchedulingRL.sub, SchedulingRL.add, 0.744, 0.905, SchedulingRL.sub, :RF, 0.74, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :EW, 0.565, SchedulingRL.add, :SLA, :JWM, SchedulingRL.sub, SchedulingRL.add, 0.691, :BSA, SchedulingRL.add, :PT, :JS, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :BSA, :JS, SchedulingRL.add, :TT, :BSA, SchedulingRL.sub, SchedulingRL.add, 0.354, 0.251, SchedulingRL.sub, 0.188, :PT]
example_tree_array_4 = Any[SchedulingRL.add, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, 0.821, 0.318, SchedulingRL.add, :RW, :JWM, SchedulingRL.sub, SchedulingRL.add, :SLA, 0.613, SchedulingRL.add, :DD, :ON, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :EW, :EW, SchedulingRL.add, :DD, :DD, SchedulingRL.add, SchedulingRL.add, :DD, 0.879, SchedulingRL.add, :TT, :ST, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, :NW, :PT, SchedulingRL.add, :CJW, :NI, SchedulingRL.add, SchedulingRL.add, :PT, :RF, SchedulingRL.sub, :ST, :ST, SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, :DD, :ON, SchedulingRL.sub, :RF, :NI, SchedulingRL.add, SchedulingRL.sub, :DD, :RW, SchedulingRL.sub, :ST, :TT]

#individuum = [example_tree_array_1]
individuum = [example_tree_array_1, example_tree_array_2, example_tree_array_3, example_tree_array_4]
expression_individdum = [get_expression(individuum[i], GP) for i in eachindex(individuum)]

actionvector = actionsfromindividuum(expression_individdum,Env_AIM)
println("Actionvector: ", actionvector, "\n")

=#

#------------------------------------------

#=
#When testing a single Feature (e.g.DD) I wanted to check weather the testagent() function for the Simple Rule without weights
# and the testGES() function for the GES with initialized weights (all 1) would return the same result, what should be the case
# but in the test they are differnt --> why do testagent() and testGES() give differnt results?

nrseeds = 30
for env in testenvs_AIM
        totalobjVal = 0
        tmpO,tmpS,tmpG = testGES(Agent_GES_AIM, ges.priorityfunctions, ges.bestweights, env, Agent_GES_AIM.objective, nrseeds, Agent_GES_AIM.rng)
        totalobjVal += tmpO
        totalobjVal /= (nrseeds*length(testenvs_AIM))
        println("Result of Rule after initializing all weights to 1: ", totalobjVal, "\n")
end
=#

#=

GP = createGP([Env_AIM],simplerules = true)
AgentGP = createagent(GP, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])

example_tree_array = build_tree(GP, true)

expr= get_expression(example_tree_array,GP)
#drawTree(expr,1)

simplifiedExpr = simplify_expression(expr,GP)
drawTree(simplifiedExpr,1)

#println("Input Expression: \n", simplifiedExpr , "\n")
#featureToPrimitives, numvalue, FeatureCounter = analyze_tree(simplifiedExpr)
#println("Feature to Primitives: \n", featureToPrimitives, "\n")
#println("Numvalue: ", numvalue, "\n")
#println("FeatureCounter: ", FeatureCounter, "\n")
#drawTree(simplifiedExpr,1)

Prio, Priostring = PriorityFunctionFromTree(simplifiedExpr,5,-5,true)
println("Priofunction Expression: ", Priostring, "\n")
println("Mask: ", Prio.nn_mask, "\n")
println("Constant: ", Prio.constant, "\n")


#priorules = Priorityfunction("TT + EW + RW + JS + DD + PT - ST")
#println("Priofunction Expression: ", priorules.expression, "\n")



#tree_array= Any[SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :TT, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, SchedulingRL.sub, 0.258, 0.325, SchedulingRL.add, :ON, :NI, SchedulingRL.add, 0.806, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.sub, :CW, nothing, nothing, :NI, nothing, nothing, SchedulingRL.sub, :RF, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, :JS, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing]
#expression= get_expression(tree_array,GP)
#println(expression)
#drawTree(expression,1)
#simplifiedExpr = simplify_expression(expression,GP)
#no_constants = reduceTreeNumericals(simplifiedExpr)
#println(no_constants)
#drawTree(no_constants,1)


#final_population= AgentGP.model[1][1]
#drawTree(get_expression(final_population[1],AgentGP.approximator))
#println(get_expression(final_population[1],AgentGP.approximator))
#println(PriorityfunctionfromTree(final_population[1],AgentGP.approximator)

#------------------------------------------

=#