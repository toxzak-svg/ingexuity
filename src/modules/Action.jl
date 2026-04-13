# ============================================================================
# Action.jl — Output Layer: execute the decided course of action
# ============================================================================
module Action

using ...Types

"""Execute the decided action based on decision + creative output"""
function execute(
    decision::Dict{Symbol, Any},
    creative::Dict{Symbol, Any},
    predictions::Vector{Prediction}
)::Dict{Symbol, Any}
    action_type = decision[:action_type]::Symbol

    if action_type == :answer
        Dict{Symbol, Any}(
            :executed => true,
            :action => "provided_answer",
            :response_style => length(creative[:framings]) > 0 ? :creative : :direct
        )
    elseif action_type == :recall
        Dict{Symbol, Any}(
            :executed => true,
            :action => "recalled_from_memory",
            :response_style => :direct
        )
    elseif action_type == :execute
        Dict{Symbol, Any}(
            :executed => true,
            :action => "executed_request",
            :response_style => :direct
        )
    else
        Dict{Symbol, Any}(
            :executed => true,
            :action => "provided_response",
            :response_style => :direct
        )
    end
end

end # module
