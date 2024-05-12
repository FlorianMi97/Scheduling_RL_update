function train!(a::Agent{GP}, generations; validationenvs=nothing, validation=false, validate_every=25, checkpoint=false, showinfo=false, showprogressbar=true, TBlog=false)
    if validate_every > 0 && !TBlog
        @warn "Eval is not stored since no TBlogger is active"
    end
    
    gp = a.approximator
    lastbest = 0 # last best value initialized

    if showprogressbar p = Progress(generations+1, 1, "Training GP") end
    if TBlog logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment) end #TODO change overwrite via kwargs? but log files are large!
    # loop for GP
    # intial Pop
    pop = []    
    
    for ngen in 0:generations #TODO parallelism?
        if ngen == 0 
            pop = initialpop(gp)
        else
            newpop = next_gen!(gp, pop, ngen)
            pop = new_pop(newpop, gp, pop, ngen)
        end
        improvement = ifelse(ngen == 0, 0 , (lastbest - pop[1][2])/lastbest)
        lastbest = pop[1][2]
        lastbesttrue = pop[1][4]
        uniquetrees = length(unique(x -> x[1], pop))
        avgage = mean([x[3] for x in pop])
        bestage = pop[1][3]

        if TBlog TBCallback(logger, pop, lastbest, lastbesttrue, improvement, uniquetrees, avgage, bestage) end

        if showprogressbar
            ProgressMeter.next!(p; showvalues = [
                                                (:generation, ngen),
                                                (:reward, lastbest),
                                                (:objective, lastbesttrue),
                                                (:improvement, improvement),
                                                ]
                                )
        end
        if showinfo
            println("best value after iteration $ngen: ", lastbest)
            println("Age is : ", [pop[i][3] for i in eachindex(pop)])
            println("unique trees: ", uniquetrees)
            println("improvement in % after iteration $ngen: ", improvement)
        end

        logging(a, ngen, lastbest, improvement, uniquetrees, avgage)

        if validate_every > 0
            if ngen % validate_every == 0 && ngen != generations
                if checkpoint && ngen != 0
                    best = validate(pop, gp, validationenvs)[1]
                    if gp.markerdensity best[end] = 0 end
                    push!(a.checkpoint, best)
                    if TBlog TBCallbackEval(logger, best[2]) end# or 4 if true objective
                elseif TBlog
                    a.model = pop
                    values = testagent(a, validationenvs)
                    TBCallbackEval(logger, values[1])
                end
            end
        end
        gp.generation +=1
    end

    if validation
        pop = validate(pop, gp, validationenvs, keep_pop=false)
        if TBlog TBCallbackEval(logger, pop[1][2]) end
    elseif !gp.stationary_samples
        pop = sortallsamples!(pop,gp)
    end

    if checkpoint
        sort!(a.checkpoint, by = x -> x[2])
        if a.checkpoint[1][2] < pop[1][2]
            insert!(pop, 1, a.checkpoint[1])
        end
    end
    a.model = pop
end

function test(a::Agent{GP},env,nrseeds)
    individuum = a.model[1][1]
    testfitness([get_expression(individuum[k],a.approximator) for k in 1:a.approximator.stages], env, a.objective, nrseeds, a.rng)
end

function nextaction(a::Agent{GP},env)
    individuum = a.model[1][1]
    actionsfromindividuum([get_expression(individuum[k],a.approximator) for k in 1:a.approximator.stages],env)[1]
end
