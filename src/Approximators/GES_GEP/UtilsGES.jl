using Plots
using ColorSchemes

mm = Plots.mm

function save_weights_boxplot(weights_log::Matrix{Float32}, priofunction, features_string::String, starting_values::Vector{Float32}, save_path::String)
    num_features = numberweights(priofunction)

    All_Features = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM", "CW", "CJW", "CT", "TT", "BSA", "DBA", "RF"]

    #sort the features in features_string in the same order as they are in All_Features
    features_string = join([f for f in All_Features if occursin(f, features_string)], " ")

    # Parse the features string to create an array of feature names
    features = split(replace(features_string, r"[\+\-]\s*" => ' '), " ", keepempty=false)

    # Prepare the data for the boxplot
    boxplot_data = [weights_log[:, weight] for weight in 1:num_features]

    # Create boxplot without including them in the legend
    p = boxplot(boxplot_data, label=nothing, title="\n Final Weight Distribution Over Runs",
        ylabel="Weight Values", palette=:nuuk10, legend=false, fontfamily="Computer Modern", fontsize=6,
        background_color_inside=:aliceblue)

    # Label the features on the x-axis
    # Modify x-axis ticks to show feature names
    xticks!(p, 1:num_features, rotation=45, font=font(6, "Arial"))
    # Label the features on the x-axis
    plot!(p, xticks=(1:num_features, features), grid=true, gridcolor=:grey0, gridlinewidth=1, size=(800, 500), margin=5mm)

    # Add markers for the initial values
    scatter!(p, 1:num_features, starting_values, label="Starting Values", marker=:diamond, color=:red, markersize=3, legend=:outerbottom)

    # Save the figure
    savefig(p, save_path)
end

# Function to create and save the diagram for results
function save_results_diagram(simulation_results, pre_run_evaluation_result::Float64, benchmark_result::Float64, save_path::String)

    # Create a boxplot for simulation results
    box_plot = boxplot([simulation_results], label=nothing, color=:royalblue, width=0.5, size=(800, 500), margin=5mm, legend=false, fontfamily="Computer Modern", fontsize=6, xticks=false)
    min_val = round(minimum(simulation_results))
    max_val = round((maximum(simulation_results)))
    mean_val = round((mean(simulation_results)))
    pre_run_evaluation_result = round(pre_run_evaluation_result)
    # Overlay data points on the boxplot
    scatter!([ones(length(simulation_results))], simulation_results,
        color=:orange, markersize=4, markerstrokecolor=:orange, label=nothing)

    # Draw a horizontal line for the pre-run evaluation and the benchmark
    hline!([pre_run_evaluation_result], label="Complex Benchmark", linestyle=:dash, color=:red, legend=:outerbottom) #label="Pre-GES Evaluation"
    hline!([benchmark_result], label="Simple Benchmark", linestyle=:dash, color=:green, legend=:outerbottom) #label="Benchmark GP Run"


    # Set labels and title
    xlabel!(box_plot, "")
    ylabel!(box_plot, "Average Objective Value")
    yticks!(box_plot, [min_val, max_val, mean_val, pre_run_evaluation_result, benchmark_result])
    title!(box_plot, "\n Objective Value Distribution Of The GP Across Runs") #Objective Value Distribution Across Runs

    # Save the figure
    savefig(box_plot, save_path)
end


function save_objective_evolution(all_rewards, pre_run_evaluation_result, save_path::String)
    num_runs = length(all_rewards)
    num_generations = length(all_rewards[1])

    p = plot(legend=:topright, size=(800, 500), margin=5mm, fontfamily="Computer Modern", fontsize=6)

    # Plot the pre-run evaluation point and connect it to the rest of the graph
    for run in 1:num_runs
        # Prepend the pre_run_evaluation_result to the current run data
        run_rewards = vcat(pre_run_evaluation_result, all_rewards[run])

        # Plot the pre-run evaluation as a separate series
        if run == 1
            scatter!(p, [0], [pre_run_evaluation_result], label="Pre-Run Evaluation", color=:black, markersize=4, markerstrokewidth=0.5)
        else
            scatter!(p, [0], [pre_run_evaluation_result], label=nothing, color=:black, markersize=4, markerstrokewidth=0.5)
        end

        # Plot the curve for the simulation run, including the pre-run evaluation
        plot!(p, 0:num_generations, run_rewards, label="Run $run", alpha=0.5, color=:auto, legend=:outertopright)
    end

    title!(p, "\n Evolution of Objective Value Over Generations")
    ticks = adjust_xticks(num_generations)
    xlabel!(p, "Generation", xticks=ticks)
    ylabel!(p, "Average Objective Value")

    # Retrieve current y-ticks ensuring all are numeric and their labels
    current_yticks, current_labels = yticks(p)[1]
    additional_ytick = round(pre_run_evaluation_result)
    new_yticks = sort(unique(vcat(current_yticks, additional_ytick)))
    new_labels = [string(y) for y in new_yticks]
    yticks!(p, new_yticks, new_labels)

    savefig(p, save_path)
end

function adjust_xticks(num_generations::Int)
    # Decide the number of ticks we want to show, based on the number of generations
    desired_num_ticks = 10

    # Calculate the spacing for ticks based on the number of generations
    tick_spacing = max(1, ceil(num_generations / desired_num_ticks))

    return 0:tick_spacing:num_generations
end

function create_and_display_rules_table(complex_rule, result_complex, simple_benchmark_rule, result_simple_benchmark, example_runs_results)
    # Create a DataFrame
    df = DataFrame(Rule=String[], Result=Float64[])

    # Add complex and simple benchmark rules
    push!(df, (Rule=complex_rule, Result=result_complex))
    push!(df, (Rule=simple_benchmark_rule, Result=result_simple_benchmark))

    # Add other runs
    for (agent, result) in example_runs_results
        expression = get_expression(agent)  # Fetch the expression for each agent
        push!(df, (Rule=expression, Result=result))
    end

    # Display the table
    pretty_table(df, header_crayon=crayon"bold cyan", row_number=true)
end

# A wrapper to handle Expr and convert it to a LaTeX string
function latexify_dispatching_rules(expr::Expr)
    # Parse the Expr to a LaTeX-friendly string
    latex_str = custom_parse(expr)

    # Convert the string to LaTeX
    latexified_expr = latexify(latex_str)

    return latexified_expr
end

function custom_parse(expr::Expr)
    # Recursively parse the expression tree and replace function calls
    if expr.head == :call
        # Map the function names to their corresponding LaTeX strings
        func_map = Dict(
            :mul => "*", :add => "+", :sub => "-",
            :squ => "^2", :max => "\\max", :min => "\\min", :ifg0 => "\\text{ifg0}"
        )

        # Extract the function from the expression
        func = expr.args[1]
        # Check if the function is in our map, otherwise convert it to string
        op = get(func_map, func, string(func))

        # Recursively parse the arguments
        args_latexified = map(custom_parse, expr.args[2:end])
        
        # Construct the LaTeX string
        return join(["(", op, " ", join(args_latexified, ", "), ")"])
    else
        # If it's not a function call, just return the expression as string
        return string(expr)
    end
end

function create_and_display_rules_table(simple_GP_results, ges_results, dispatching_rules, initial_weights, save_path::String)
    # Create a DataFrame
    df = DataFrame(Rule = Int[], GP_Reward = Float64[], GES_Reward = Float64[], Dispatching_Rule = String[], Initial_Weights = Vector{Float64}[])

    # Populate the DataFrame
    for i in eachindex(simple_GP_results)
        gp_reward = simple_GP_results[i][1]  
        ges_reward = ges_results[i][1]       
        dispatching_rule = dispatching_rules[i][2]
        initial_weight = initial_weights[i][2]
        

        # Add to DataFrame
        push!(df, (Rule = i, GP_Reward = gp_reward, GES_Reward = ges_reward, Dispatching_Rule = dispatching_rule, Initial_Weights = initial_weight))
    end

    # Save the DataFrame as an HTML file
    Save_as_html_content(df, ["Rule", "GP Reward", "GES Reward", "Dispatching Rule", "Initial Weights"], save_path)
    
end

function Save_as_html_content(df, Headers::Vector{String}, save_path::String, priomatrix = false)

    if priomatrix == true
        n, m = size(df)
        df_full = DataFrame()
        job_header = ["Jobs \\ Machines"; ["$(i-1)" for i in 2:n+1]]
        insertcols!(df_full, 1, :Job => job_header)
        for j in 1:m
            col_header = ["$(j)"; df[:, j]]
            insertcols!(df_full, ncol(df_full)+1, Symbol("Machine_$j") => col_header)
        end

        io = IOBuffer()
        pretty_table(
            io, 
            df_full,
            show_header = false, 
            alignment = :l,
            backend = Val(:html), 
            tf = tf_html_minimalist )
    
        # Write the HTML content to a file
        html_content = String(take!(io))  # Retrieve the string content from IOBuffer
        open(save_path, "w") do file
            write(file, html_content)
        println("Table saved to: $save_path")
        end 
    
    else  
        io = IOBuffer()
        pretty_table(
            io, 
            df, 
            header = Headers, 
            backend = Val(:html), 
            tf = tf_html_minimalist )
    
        # Write the HTML content to a file
        html_content = String(take!(io))  # Retrieve the string content from IOBuffer
        open(save_path, "w") do file
            write(file, html_content)
        println("Table saved to: $save_path")
        end
    end 
end

