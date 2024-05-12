
function logging(a::Agent{GES}, elapsed_time_gen, weights, lastbest, improvement, Adam_state=nothing)
    haskey(a.logger, "elapsed_time_per_gen") || (a.logger["elapsed_time_per_gen"] = [])
    push!(a.logger["elapsed_time_per_gen"], elapsed_time_gen)  

    haskey(a.logger, "reward") || (a.logger["reward"] = [])
    push!(a.logger["reward"], lastbest)  
    
    haskey(a.logger, "improvement") || (a.logger["improvement"] = [])
    push!(a.logger["improvement"], improvement)  

    haskey(a.logger, "weights") || (a.logger["weights"] = [])
    push!(a.logger["weights"], weights)
    
    

    if Adam_state !== nothing
        haskey(a.logger, "ADAM_state") || (a.logger["ADAM_state"] = [])
        push!(a.logger["ADAM_state"], Adam_state) 
    end
end

function TBCallback(a::Agent{GES}, logger, weights, lastbest, improvement)
    
    

    with_logger(logger) do
        
        @info "training" avg_reward = -lastbest # TODO add best and worst as well? from samples best_reward = maximum(wr) worst_reward = minimum(wr)
        @info "metrics" improvement = improvement
        @info "weights" current_weights = weights 
        
       
    end
end

function TBCallbackEval(a::Agent{GES},logger, avgreward, gap)
    with_logger(logger) do
        @info  "eval" avg_objective = avgreward gap = gap
    end
end
