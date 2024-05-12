
logging(a::Agent{GP}, ngen, lastbest, improvement, uniquetrees, avgage) = begin
    append!(a.logger["reward"], lastbest)

    if ngen == 0
        a.logger["improvement"] = [0]
        a.logger["unique_trees"] = [uniquetrees]
        a.logger["average_age"] = [avgage]
    else
        append!(a.logger["improvement"], improvement)
        append!(a.logger["unique_trees"], uniquetrees)
        append!(a.logger["average_age"], avgage)
    end
end

function TBCallback(logger, population, lastbest, lastbesttrue, improvement, uniquetrees, avgage, bestage)
    densitymax = maximum(x -> x[6] ,population)
    densitybest = population[1][6]
    sizebest = sizeofrule(population[1][1][1])                              
    with_logger(logger) do
        
        @info "training" avg_reward = -lastbest avg_true_objective = -lastbesttrue improvement = improvement
        @info "metrics" number_uniquetrees = uniquetrees average_age = avgage age_best = bestage density_best = densitybest density_max = densitymax size_best = sizebest log_step_increment=0
        
        #TODO add more!!!!
        # e.g. show best tree as infix?
        # population mean fitness
        # ...
    end
end

function TBCallbackEval(logger, avgreward)
    with_logger(logger) do
        @info  "eval" avg_objective = avgreward log_step_increment=0
    end
end