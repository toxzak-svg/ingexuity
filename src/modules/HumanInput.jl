# ============================================================================
# HumanInput.jl — Parse and validate user input
# ============================================================================
module HumanInput

using Dates
using ..Types: HumanInput as HumanInputType

export process

function process(raw::String; session_id::Int64=0)
    HumanInputType(raw, Dates.now(), session_id)
end

end # module
