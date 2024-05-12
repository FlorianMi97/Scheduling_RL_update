using Plots
using StatsPlots
using SchedulingRL
using Hyperopt
using Optim: optimize, Options, minimum, minimizer, NelderMead
using ProgressMeter
using Distributed

#=
if nworkers() == 1 
    addprocs(4) 
end
@everywhere using Hyperopt
@everywhere using SchedulingRL
@everywhere using Distributed
@everywhere using Optim: optimize, Options, minimum, minimizer, NelderMead


@everywhere f(a;c=10) = sum(@. 100 + (a-3)^2 + (c-100)^2)

bohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])), a = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
    if state !== nothing
        a,c = state
    end
    res = optimize(x->f(x[1],c=x[2]), [a,c], NelderMead(), Options(f_calls_limit=round(Int, i)))
    minimum(res), minimizer(res)
end

@everywhere println("Worker $(myid()) is alive and well!")

=#
instance = "useCase_2_stages"
Env_AIM_b = createenv(as="AIM", instanceType="usecase", instanceName=instance)
testenvs_b = [Dict("layout" => "Flow_shops",
    "instancetype" => "usecase",
    "instancesettings" => "base_setting",
    "datatype" => "data_ii",
    "instancename" => instance)]

testenvs_AIM = generatetestset(testenvs_b, 100, actionspace="AIM")
priofunction_benchmark = Priorityfunction("CJW + NW + DBA + ST + DD + ON + NI")
Agent_GES = createagent(createGES(
                    "global", [Env_AIM_b], priofunction_benchmark, optimizer="ADAM"),
                "AIM", obj=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
result_benchmark= testagent(Agent_GES, testenvs_AIM)



bohb = @hyperopt for i = 500, sampler = Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])),#
    α_μ = LinRange(1e-3, 0.2, 20),
    α_σ = LinRange(1e-3, 0.2, 20)
    #ngen = LinRange(50, 500, 10)
    #learningrates_decay_μ = LinRange(0.5, 0.99, 20),
    #learningrates_decay_σ = LinRange(0.5, 0.99, 20),
    #decayStep = LinRange(1, 10, 10)
    #beta1 = LinRange(0.8, 0.99, 10),
    #beta2 = LinRange(0.9, 0.999, 10)
    begin
        if state !== nothing
            α_μ, α_σ = state #learningrates_decay_μ,learningrates_decay_σ,decayStep   
        end

        instance = "useCase_2_stages"
        Env_AIM = createenv(as="AIM", instanceType="usecase", instanceName=instance)
        testenvs = [Dict("layout" => "Flow_shops",
            "instancetype" => "usecase",
            "instancesettings" => "base_setting",
            "datatype" => "data_ii",
            "instancename" => instance)]

        testenvs_AIM = generatetestset(testenvs, 100, actionspace="AIM")
        priofunction = Priorityfunction("CJW + NW + DBA + ST + DD + ON + NI")

        #number_of_generations = round(Int, ngen)
        #decayStep_size = max(round(Int, decayStep), 1)

        res = trainagent!(createagent(createGES(
                    "global", [Env_AIM], priofunction,
                    optimizer="ADAM",
                    learningrates_ADAM=[α_μ, α_σ],
                    #adam_params = Dict("β1" => beta1, "β2" => beta2, "ϵ" => 1e-8)
                    #learningrates_SGD = [α_μ, α_σ],
                    #learningrates_decay_SGD = [learningrates_decay_μ, learningrates_decay_σ],
                    #decayStepSize = decayStep_size
                ),
                "AIM", obj=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            generations=250,
            evalevery=1,
            finaleval=true,
            testenvs=testenvs_AIM)[1]


        res, (α_μ, α_σ) #learningrates_decay_μ,learningrates_decay_σ,decayStep )
    end
end

plot(bohb)
savefig("C:\\Users\\floppy\\Desktop\\Hyperparameter Tuning\\Hyperopt.png")
print(bohb)


#=
@everywhere function objective(params; resources)


    α_μ, α_σ, beta1, beta2 = params
    instance = "useCase_2_stages"
    Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
    testenvs = [Dict("layout" => "Flow_shops" ,
        "instancetype" => "usecase" ,
        "instancesettings" => "base_setting",
        "datatype" => "data_ii",
        "instancename" =>instance)]

    testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
    priofunction = Priorityfunction("PT + DD + RW + JS + EW + TT + BSA")

    numer_of_generations = round(Int, resources)

    res = trainagent!(createagent(createGES(
        "global", [Env_AIM], priofunction,
        optimizer = "ADAM",
        learningrates_ADAM = [α_μ, α_σ],
        adam_params = Dict("β1" => beta1, "β2" => beta2, "ϵ" => 1e-8)),
        "AIM", obj = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 
        generations = numer_of_generations,
        evalevery = 0, 
        finaleval = true, 
        testenvs = testenvs_AIM)[1]
    res,  (α_μ, α_σ, beta1, beta2)
end
candidates = (α_μ = LinRange(1e-5, 1e-1, 10),
α_σ = LinRange(1e-5, 1e-1, 10),
beta1 = LinRange(0.8, 0.99, 10),
beta2 = LinRange(0.9, 0.999, 10))

hohb = hyperband(objective, candidates, R=50, η=3)
=#
