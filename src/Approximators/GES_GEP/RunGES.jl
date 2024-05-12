using ProgressMeter

function train!(a::Agent{GES}, generations, evalevery, testenvs, finaleval; showinfo=false, showprogressbar=true, TBlog=false, termination_criteria =100)

    #if evalevery > 0 && !TBlog
    #    @warn "Eval is not stored since no TBlogger is active"
    #end
    ges = a.approximator
    initial_value = testagent(a, testenvs)
    #println("totalobjVal before run: ", initial_value[1], "\n")
    lastbest = initial_value[1]
    best_value = 1e4
    Current_results = initial_value

    #println("pre-GES evaluation result: ", lastbest)

    lastweights = ges.bestweights
    update_weights_with_signs!(a.approximator, lastweights, a.approximator.priorityfunctions)
    last_sigma = ges.sigma

    improvement = 0
    no_improvement_count = 0
    explode_condition = false
    higher_count = 0

    for env in ges.envs # reset envs before starting training
        resetenv!(env)
    end

    if showprogressbar
        p = Progress(generations, 1, "Training Weights with GES")
    end
    if TBlog
        logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment)
    end #TODO change overwrite via kwargs? but log files are large!
    # loop for GEP
    for ngen in 1:generations

        ges.generation += 1
        start_time_gen = now(tz"GMT")

        μ, σ, mean_simulation_results_plus, mean_simulation_results_minus = update!(ges, ngen)
        update_weights_with_signs!(a.approximator, [μ], a.approximator.priorityfunctions)

        elapsed_time_gen = now(tz"GMT") - start_time_gen

        # Save last values and check for NaNs # TODO non global
        for i in eachindex(μ[1])
            if isnan(μ[1][i])
                println("μ[i] was NaN for weight ", i, "in generation ", ngen, " and will be resetset to its previous value")
                μ[1][i] = lastweights[1][i]
            else
                lastweights[1][i] = μ[1][i]
            end
            if isnan(σ[1][i])
                println("σ[i] was NaN for weight ", i, "in generation ", ngen, " and will be resetset to its previous value")
                σ[1][i] = last_sigma[1][i]
            else
                last_sigma[1][i] = σ[1][i]
            end
        end

        # TODO define values to log and logger function
        if TBlog
            TBCallback(a, logger, μ, lastbest, improvement)
        end



        if showprogressbar
            ProgressMeter.next!(p; showvalues=[
                (:weights_current_generation, ges.weights_with_signs),
                (:minimum, best_value),
                (:last_value, lastbest),
                (:improvement_percentage, improvement)
            ])
        end


        if showinfo
            println("\n", "CURRENT GENERATION: ", ngen, "\n")
            println("Weights in current generation: ", ges.weights_with_signs)
            #println("Sigma in current generation: ", ges.sigma)
            println("best value after iteration $ngen: ", lastbest)
            println("improvement in % after iteration $ngen: ", improvement)
            println("Average simulation results were: ", "f_w⁺: ", mean_simulation_results_plus,
                " and f_w⁻ ", mean_simulation_results_minus)
        end

        if evalevery > 0
            if ngen % evalevery == 0
                a.model = [ges.bestweights, ges.priorityfunctions] #TODO non global
                #update_weights_with_signs!(a.approximator, [μ], a.approximator.priorityfunctions)
                Current_results = testagent(a, testenvs)
                improvement = ((lastbest - Current_results[1]) / lastbest) * 100
                lastbest = Current_results[1]
                if Current_results[1] ≤ best_value
                    higher_count = 0
                    if Current_results[1] < best_value
                        best_value = Current_results[1]
                        no_improvement_count = 0
                    end
                else
                    no_improvement_count += 1
                    if no_improvement_count >= termination_criteria
                        a.model = [ges.bestweights, ges.priorityfunctions]
                        #update_weights_with_signs!(a.approximator, [μ], a.approximator.priorityfunctions)
                        println("training aborted due stopping criteria in gen: $ngen")
                        break
                    end
                    if Current_results[1] > best_value * 5
                        explode_condition = true
                        a.model = [ges.bestweights, ges.priorityfunctions]
                        #update_weights_with_signs!(a.approximator, [μ], a.approximator.priorityfunctions)
                        println("training aborted due to explode in loss function in gen: $ngen")
                        break
                    end
                    if Current_results[1] > best_value
                        higher_count += 1
                        if higher_count >= 50
                            a.model = [ges.bestweights, ges.priorityfunctions]
                            #update_weights_with_signs!(a.approximator, [μ], a.approximator.priorityfunctions)
                            println("training aborted due to to high objective value for more than 50 generations")
                            break
                        end
                    end
                end
                if TBlog
                    TBCallbackEval(a, logger, Current_results[1], Current_results[6])
                end
            end
        end

        if ges.optimizer == "ADAM"
            logging(a, elapsed_time_gen, ges.weights_with_signs, lastbest, improvement, ges.adam_state)
        else
            logging(a, elapsed_time_gen, ges.weights_with_signs, lastbest, improvement)
        end
        


    end
    update_weights_with_signs!(a.approximator, [ges.bestweights], a.approximator.priorityfunctions)
    a.model = [ges.bestweights, ges.priorityfunctions]
    #TODO non global

    if finaleval

        values = testagent(a, testenvs)
        if TBlog
            TBCallbackEval(a, logger, values[1], values[6])
        end
        return values
        #if explode_condition # To terminate hyperparameter testing if the loss function explodes --> remove again
        #   return [1e4]
        #elseif higher_count >= 30 # To terminate hyperparameter testing if the loss function is too high --> remove again
        #return [1e4]
        #else
        #values = testagent(a, testenvs)
        #if TBlog TBCallbackEval(a, logger, values[1], values[6]) end
        #return values
        #end

    end
end

function update!(ges::GES, ngen)

    if !ges.stationary_samples
        if ngen % ges.change_every == 0
            if ges.sample_cycling
                ges.sample_tens_digit = (ngen ÷ ges.change_every) % (ges.steps_in_cycle)
            else
                setsamples!.(ges.envs, ges.rng, nSamples=ges.samples_generation)
            end
        end
    end

    if ges.optimizer == "GradientDecent"
        structured_μ, structured_σ, m_plus, m_minus = gradientdecent!(ges)
    elseif ges.optimizer == "ADAM"
        structured_μ, structured_σ, m_plus, m_minus = adamoptimizer!(ges)
    else
        error("Optimizer not supported")
    end

    return structured_μ, structured_σ, m_plus, m_minus

end


function gradientdecent!(ges::GES)

    μ = vcat(ges.bestweights...)
    σ = vcat(ges.sigma...)
    α_μ = ges.learningrates_SGD[1]  # Define learning rate for μ
    α_σ = ges.learningrates_SGD[2]  # Define learning rate for σ

    # Initialize momentum state if not already initialized
    if !isdefined(ges, :nesterov_state) || isempty(ges.nesterov_state)
        ges.nesterov_state = Dict("m_μ" => zeros(length(μ)), "m_σ" => zeros(length(σ)))
    end
    β_μ, β_σ = ges.SGD_params["βμ"], ges.SGD_params["βσ"]
    m_μ, m_σ = ges.nesterov_state["m_μ"], ges.nesterov_state["m_σ"]

    # Apply interim momentum-based parameter update (Nesterov momentum)
    interim_μ = μ + β_μ .* m_μ
    interim_σ = σ + β_σ .* m_σ

    weights_pos, weights_neg, ε_values = generate_perturbations(interim_μ, interim_σ, ges)
    f_w⁺, f_w⁻, m_plus, m_minus = evaluation_simulation(weights_pos, weights_neg, ges)
    f_w⁺_N, f_w⁻_N = normalize_performance(f_w⁺, f_w⁻)
    Δμ, Δσ = estimate_gradients(ε_values, f_w⁺_N, f_w⁻_N, interim_μ, interim_σ, ges)

    # Update momentum based on gradients at interim parameters
    m_μ = β_μ .* m_μ + α_μ .* Δμ
    m_σ = β_σ .* m_σ + α_σ .* Δσ

    # Update parameters μ and σ
    for i in eachindex(μ)
        μ[i] += m_μ[i]
        σ[i] += m_σ[i]
    end

    # Update the GES structure
    structured_μ = segment_weights(μ, ges)
    structured_σ = segment_weights(σ, ges)
    ges.bestweights = structured_μ
    ges.sigma = structured_σ

    # Decay learning rates
    if ges.decay_SGD && ges.generation % ges.decayStepSize == 0
        ges.learningrates_SGD[1] *= ges.learningrates_decay_SGD[1]
        ges.learningrates_SGD[2] *= ges.learningrates_decay_SGD[2]
    end

    return structured_μ, structured_σ, m_plus, m_minus
end

#helper function to segment weights again
function segment_weights(flat_weights, ges::GES)
    segmented_weights = []
    offset = 1
    for stage_index in 1:length(ges.priorityfunctions)
        stage_length = length(ges.bestweights[stage_index])
        stage_weights = flat_weights[offset:(offset+stage_length-1)]
        push!(segmented_weights, stage_weights)
        offset += stage_length
    end
    return segmented_weights
end

function generate_perturbations(μ, σ, ges::GES)

    N = Int(ges.popSize / 2) # Number of perturbations to evaluate
    D = length(μ) # Number of weights

    weights_pos = []
    weights_neg = []
    ε_values = []
    for n = 1:N  # Loop over the number of pertubation (!= samples)
        weights_p = []
        weights_n = []
        ε_val = []
        for i = 1:D
            # Draw perturbation ε ~ N(0, σ^2)
            if σ[i] == 0
                σ[i] = 1e-8
            elseif σ[i] > 1e8
                σ[i] = 1e8
            elseif isnan(σ[i])
                println("σ[i] was NaN for sample ", n, " and weight ", i)
                σ[i] = 1e-8
            end
            ε = randn() * σ[i]
            w⁺ = μ[i] .+ ε
            w⁻ = μ[i] .- ε
            append!(weights_p, w⁺)
            append!(weights_n, w⁻)
            append!(ε_val, ε)
            #println("Perturbation ε[", i, "]: ", ε, " | w⁺: ", w⁺, " | w⁻: ", w⁻) 
        end
        push!(weights_pos, weights_p)
        push!(weights_neg, weights_n)
        push!(ε_values, ε_val)
    end
    return weights_pos, weights_neg, ε_values
end

function evaluation_simulation(weights_pos, weights_neg, ges::GES)

    N = Int(ges.popSize / 2) # Number of perturbations to evaluate

    f_w⁺ = Vector{Float64}(undef, N)
    f_w⁻ = Vector{Float64}(undef, N)
    Threads.@threads for n = 1:N # Loop over the number of pertubation (!= samples)

        ges_copy = deepcopy(ges)
        weights_p = weights_pos[n]
        weights_n = weights_neg[n]

        # Segment the weights for positive and negative perturbation according to stages
        segmented_weights_pos = segment_weights(weights_p, ges_copy)
        segmented_weights_neg = segment_weights(weights_n, ges_copy)
        #println("Segmented weights for w⁺[", n, "]: ", segmented_weights_pos)

        #fitness values over all samples_generations 
        _, _, f_w_plus = detFitness(ges_copy.priorityfunctions, segmented_weights_pos, ges_copy) #TODO adapt for non global GES
        _, _, f_w_minus = detFitness(ges_copy.priorityfunctions, segmented_weights_neg, ges_copy) #TODO adapt for non global GES
        f_w⁺[n] = f_w_plus
        f_w⁻[n] = f_w_minus
    end
    return f_w⁺, f_w⁻, mean(f_w⁺), mean(f_w⁻)
end

function normalize_performance(f_w⁺, f_w⁻)
    f_w⁺_N = []
    f_w⁻_N = []
    std_f_w⁺ = std(f_w⁺)
    std_f_w⁻ = std(f_w⁻)
    for i in eachindex(f_w⁺)
        if std_f_w⁺ != 0
            push!(f_w⁺_N, (f_w⁺[i] - mean(f_w⁺)) / std_f_w⁺)
        else
            push!(f_w⁺_N, 0.0)
        end

        if std_f_w⁻ != 0
            push!(f_w⁻_N, (f_w⁻[i] - mean(f_w⁻)) / std_f_w⁻)
        else
            push!(f_w⁻_N, 0.0)
        end
    end
    return f_w⁺_N, f_w⁻_N
end

function estimate_gradients(ε_values, f_w⁺_N, f_w⁻_N, μ, σ, ges::GES)

    N = Int(ges.popSize / 2) # Number of perturbations to evaluate
    D = length(μ) # Number of weights
    ϵ = 1e-8  # Small constant to prevent division by zero

    # Initialize Δμ and Δσ
    Δμ = zeros(D)
    Δσ = zeros(D)

    for i = 1:D
        Δμ_i = 0.0
        Δσ_i = 0.0
        for n = 1:N
            σ_i = σ[i] != 0 ? σ[i] : ϵ
            Δμ_i += (ε_values[n][i] / σ_i) * ((f_w⁺_N[n]-f_w⁻_N[n])[1] / 2)
            Δσ_i += ((ε_values[n][i]^2 - σ_i) / σ_i) * ((f_w⁺_N[n]-f_w⁻_N[n])[1] / 2)
        end
        Δμ[i] = Δμ_i / N
        Δσ[i] = Δσ_i / N
    end
    return Δμ, Δσ
end




function adamoptimizer!(ges::GES)

    α_μ = ges.learningrates_ADAM[1]  # Define learning rate for μ
    α_σ = ges.learningrates_ADAM[2]  # Define learning rate for σ

    # Initialize ADAM state if not already initialized
    if !isdefined(ges, :adam_state) || isempty(ges.adam_state)
        initialize_adam_state!(ges)
    end

    # Correct extraction of ADAM state
    m_μ, v_μ, m_σ, v_σ, t = ges.adam_state["m_μ"], ges.adam_state["v_μ"], ges.adam_state["m_σ"], ges.adam_state["v_σ"], ges.adam_state["t"]

    μ = vcat(ges.bestweights...)
    σ = vcat(ges.sigma...)

    # Rest of the optimization steps
    weights_pos, weights_neg, ε_values = generate_perturbations(μ, σ, ges)
    f_w⁺, f_w⁻, m_plus, m_minus = evaluation_simulation(weights_pos, weights_neg, ges)
    f_w⁺_N, f_w⁻_N = normalize_performance(f_w⁺, f_w⁻)
    Δμ, Δσ = estimate_gradients(ε_values, f_w⁺_N, f_w⁻_N, μ, σ, ges)

    # Apply ADAM update and get the updated state
    μ, σ, m_μ, v_μ, m_σ, v_σ, t = adam_update!(ges, μ, σ, Δμ, Δσ, (m_μ, v_μ, m_σ, v_σ, t), α_μ, α_σ)

    # Correctly store updated ADAM state
    ges.adam_state = Dict("m_μ" => m_μ, "v_μ" => v_μ, "m_σ" => m_σ, "v_σ" => v_σ, "t" => t)


    # Update the GES structure
    structured_μ = segment_weights(μ, ges)
    structured_σ = segment_weights(σ, ges)
    ges.bestweights = structured_μ
    ges.sigma = structured_σ

    if ges.decay_ADAM && ges.generation % ges.decayStepSize == 0
        ges.learningrates_ADAM[1] *= ges.learningrates_decay_ADAM[1]
        ges.learningrates_ADAM[2] *= ges.learningrates_decay_ADAM[2]
    end

    return structured_μ, structured_σ, m_plus, m_minus

end

function initialize_adam_state!(ges::GES)
    #TODO adapt for non global GES
    ges.adam_state = Dict(
        "m_μ" => zeros(length(ges.bestweights[1])),
        "v_μ" => zeros(length(ges.bestweights[1])),
        "m_σ" => zeros(length(ges.sigma[1])),
        "v_σ" => zeros(length(ges.sigma[1])),
        "t" => 0
    )
end

function adam_update!(ges, μ, σ, Δμ, Δσ, adam_state, α_μ, α_σ)

    β1, β2, ϵ = ges.adam_params["β1"], ges.adam_params["β2"], ges.adam_params["ϵ"]
    m_μ, v_μ, m_σ, v_σ, t = adam_state

    # Update moments for μ and σ
    m_μ .= β1 .* m_μ .+ (1.0 - β1) .* Δμ
    v_μ .= β2 .* v_μ .+ (1.0 - β2) .* (Δμ .^ 2)
    m_σ .= β1 .* m_σ .+ (1.0 - β1) .* Δσ
    v_σ .= β2 .* v_σ .+ (1.0 - β2) .* (Δσ .^ 2)

    # Increment iteration count
    t += 1

    # Compute bias-corrected moments
    m_μ_hat = m_μ ./ (1.0 - β1^t)
    v_μ_hat = max.(v_μ ./ (1.0 - β2^t), 0)
    m_σ_hat = m_σ ./ (1.0 - β1^t)
    v_σ_hat = max.(v_σ ./ (1.0 - β2^t), 0)

    # Update parameters
    μ .+= α_μ .* m_μ_hat ./ (sqrt.(v_μ_hat) .+ ϵ)
    σ .+= α_σ .* m_σ_hat ./ (sqrt.(v_σ_hat) .+ ϵ)

    return (μ, σ, m_μ, v_μ, m_σ, v_σ, t)
end

function evalGESsample(priorityfunctions::Vector{Priorityfunction}, weights, env, objective, pointer, rng, type="greedy")
    """
    Evaluates an individual's fitness based on a single sample or scenario
    """
    metrics = [0, 0, 0, 0, 0, 0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo, pointer)
    t = false
    if type == "greedy"
        state = flatstate(env.state)
        while !t
            action = Base.invokelatest(translateaction, priorityfunctions, weights, env)  #translateaction(priorityfunctions, weights, env) 
            nextstate, rewards, t, info = step!(env, action, rng)
            metrics += rewards
        end
    else
        error("Other than 'Greedy' not implemented yet")
    end
    fitness = dot(objective, metrics)
    return fitness, metrics
end

function evalGESfitness(priorityfunctions::Vector{Priorityfunction}, weights, env, objective, nrsamples, rng, sample_tens_digit; type="greedy")
    """
    aggregates the fitness evaluations from multiple samples to determine an individual's overall fitness
    """
    metrics = []
    fitness = []

    for i in 1:nrsamples

        pointer = isempty(env.samples) ? nothing : (i + nrsamples * sample_tens_digit)
        tmpfitness, tmpmetrics = evalGESsample(priorityfunctions, weights, env, objective, pointer, rng, type)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
    end

    return sum(fitness), fitness, metrics
end

function detFitness(priorityfunctions::Vector{Priorityfunction}, weights, ges::GES)
    """
    Determines the fitness of an individual potentially across multiple environments
    """
    fitness = 0
    for e in eachindex(ges.envs)
        tmpF, tmpO = evalGESfitness(priorityfunctions, weights, ges.envs[e], ges.objective, ges.samples_generation, ges.rng, ges.sample_tens_digit)
        fitness += (tmpF / ges.samples_generation)
    end
    fitness /= length(ges.envs)

    #if gep.markerdensity
    #    Ftuple = tuple([prio,weights, fitness,  get_marker(indi,gp)[1], 0]) # NOT YET IMPELEMENTED FOR GES 
    #else
    Ftuple = tuple(priorityfunctions, weights, fitness)
    #end
    return Ftuple
end


function test(a::Agent{GES}, env, nrseeds)
    if a.approximator.granularity == "global"

        testGES(a, a.approximator.priorityfunctions, a.approximator.bestweights, env, a.objective, nrseeds, a.rng)
    else
        error("non global granularity not implemented yet")
    end
end

function testGES(a, priorityfunctions::Vector{Priorityfunction}, weights, env, objective, nrsamples, rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    if a.approximator.granularity == "global"

        for i in 1:nrsamples
            isempty(env.samples) ? pointer = nothing : pointer = i
            tmpfitness, tmpmetrics = evalGESsample(priorityfunctions, weights, env, objective, pointer, rng)
            append!(metrics, tmpmetrics)
            append!(fitness, tmpfitness)

            if env.type == "usecase"
                if objective == [-1.0, -0.0, -0.0, -0.0, -0.0, -0.0]
                    piobjective = env.samples[pointer]["objective_[1, 0]"]
                    append!(gaps, (tmpfitness / piobjective) - 1)
                else
                    piobjective = env.samples[pointer]["objective_[0, 1]"]
                    append!(gaps, (tmpfitness - piobjective))
                end
            else
                piobjective = env.samples[pointer]["objective"]
                append!(gaps, (tmpfitness / piobjective) - 1)
            end
        end

    else
        error("non global granularity not implemented yet")
    end
    return sum(fitness), fitness, gaps
end

function nextaction(a::Agent{GES}, env)
    # TODO has to be adapted for non global GES
    translateaction(a.model[1], env, a.model[2])
end

function update_weights_with_signs!(ges, current_weights, priorityfunctions::Vector{Priorityfunction})
    if ges.granularity == "global"
        #println("current weights in generation $(ges.generation): ", current_weights[1])
        # Initialize a container for weights with applied signs, ensuring it matches the type of current_weights
        if ges.generation == 0
            weights_with_signs = deepcopy(current_weights[1])
        else 
            weights_with_signs = deepcopy(current_weights[1][1])
        end
        
        # Extract the expression string from the first priority function.
        expression = priorityfunctions[1].expressionString
        #println("expression: ", expression)
        features = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM", "CW", "CJW", "CT", "TT", "BSA", "DBA", "RF"]

        # Initialize the signs array with a specified type to ensure type stability
        signs = Vector{Int}(undef, length(features))

        # Extract the signs from the expression string for all features present in the expression string. Example of such a expression string: "w.ω_DD * f.DD - w.ω_ON * f.ON + w.ω_ST * f.ST + w.ω_NW * f.NW + w.ω_EW * f.EW + w.ω_CW * f.CW - w.ω_CT * f.CT + w.ω_DBA * f.DBA"
        signs = []
        for feature in features
            if occursin(feature, expression)
                if occursin("- w.ω_$feature", expression)
                    push!(signs, -1)
                else
                    push!(signs, 1)
                end
            end
        end
        #println("signs: ", signs)
        # update the weights with the signs
        for i in eachindex(weights_with_signs)
            weights_with_signs[i] *= signs[i]
        end

        #println("weights with signs: ", weights_with_signs, "\n")

        # Update the GES structure with the calculated weights with signs.
        ges.weights_with_signs[1] = weights_with_signs
    else
        error("non-global granularity not implemented yet")
    end
end

