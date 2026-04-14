# ============================================================================
# Output.jl — Render the final output
# ============================================================================
module Output

using ..Types: Output as OutputType, Response as ResponseType
using Dates

export render

function render(response::ResponseType, comprehension; voice_enabled::Bool=false)
    OutputType(response.content, voice_enabled, Dates.now())
end

end # module
