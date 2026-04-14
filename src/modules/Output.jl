# ============================================================================
# Output.jl — Render the final output
# ============================================================================
module Output

export render

function render(response::Response, comprehension; voice_enabled::Bool=false)
    Output(response.content, voice_enabled, now())
end

end # module
