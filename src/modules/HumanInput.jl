# ============================================================================
# HumanInput.jl — Parse and validate user input
# ============================================================================
module HumanInput

export process

"""Process raw user input into a HumanInput struct"""
function process(raw::String; session_id::Int64=0)
    # Keep it simple — no complex parsing needed for v1
    HumanInput(raw, now(), session_id)
end

end # module
