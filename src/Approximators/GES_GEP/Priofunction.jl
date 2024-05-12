struct Priorityfunction
    expression
    expressionString::String
    nn_mask::Vector{Bool}
    max::Vector{Float32}
    min::Vector{Float32}
    constant::Union{Float32,Nothing} # Optional field to store a numerical constant
end

function Priorityfunction(s::String, 
    max::Union{Vector{Number},Number}=100,
    min::Union{Vector{Number},Number}=-100,
    constant::Union{Float32,Nothing}=nothing)

    #=

    features = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM", "CW", "CJW", "CT", "TT", "BSA", "DBA", "RF"]
    mask = [false for _ in features]

    # Construct the expression string with placeholders for weights and features
    expr_parts = []
    for feature in features
        if occursin(feature, s)
            push!(expr_parts, "(w.ω_$feature * f.$feature)")
            mask[findfirst(feature -> feature == feature, features)] = true
        end
    end

    # Join all parts with '+', assuming s includes all features as additive
    expression_string = join(expr_parts, " + ")
    println("Expression string generated in Priorityfunction: ", expression_string)

    # Create the final expression as a function
    full_expression = "(w, f) -> " * expression_string
    println("Full expression for Meta.parse: ", full_expression)
    expr = eval(Meta.parse(full_expression))

    =#

    
    features = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM","CW", "CJW", "CT","TT", "BSA", "DBA", "RF"]
    mask = [true for _ in features]

    for i in eachindex(features)
        s = replace(s, features[i] => "w.ω_" * features[i] * " * f." * features[i])
        mask[i] = occursin(features[i],s)
    end

    f = Meta.parse("(w,f) -> " * s)
    expr = eval(f)
    

    if max isa Number
        max = [max for i in mask if i]
    end
    if min isa Number
        min = [min for i in mask if i]
    end
    if sum(mask) == 0
        error("no features selected")
    end
    if max isa Vector{Number}
        if length(max) != sum(mask)
            error("max vector has wrong length")
        end
    end
    if min isa Vector{Number}
        if length(min) != sum(mask)
            error("min vector has wrong length")
        end
    end

    Priorityfunction(expr, s, mask, max, min, constant)
end


function PriorityFunctionFromTree(expr,
    max::Union{Vector{Number},Number}=100,
    min::Union{Vector{Number},Number}=-100,
    keep_numerical_nodes::Bool=true, kwargs...)

    featureToPrimitives, numvalue, _ = analyze_tree(expr)

    All_Features = [:PT, :DD, :RO, :RW, :ON, :JS, :ST, :NI, :NW, :SLA, :EW, :JWM, :CW, :CJW, :CT, :TT, :BSA, :DBA, :RF]
    operations = []
    normalizedOperations = []

    # Create dictionaries to index features by their original order as symbols
    orderedFeatures = Dict(f => index for (index, f) in enumerate(All_Features))

    # Initialize a list for holding the operations with their index
    for feature in All_Features
        if haskey(featureToPrimitives, feature)
            operation = featureToPrimitives[feature]
            featureString = "$feature"
            opSign = operation == :sub ? "- " : "+ "
            push!(operations, (orderedFeatures[feature], "$opSign$featureString"))
            push!(normalizedOperations, (orderedFeatures[feature], "+$featureString"))  # All ops as +
        end
    end

    # Sort the operations based on the feature order in All_Features
    sort!(operations, by=x -> x[1])
    sort!(normalizedOperations, by=x -> x[1])

    # Generate the strings of operations
    operationString = join([op[2] for op in operations], " ")
    normalizedOperationString = join([op[2] for op in normalizedOperations], " ")

    # Remove the leading "+" if present and clean up spaces
    operationString = String(strip(lstrip(operationString, '+')))
    normalizedOperationString = String(strip(lstrip(normalizedOperationString, '+')))

    #println("Operation String generated from the tree: ", operationString)
    #println("Normalized Operation String generated from the tree: ", normalizedOperationString, "\n")

    # Handle the constant
    constant = keep_numerical_nodes ? Float32(numvalue) : nothing

    return Priorityfunction(operationString, max, min, constant), operationString
end








