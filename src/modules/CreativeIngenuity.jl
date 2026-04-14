# ============================================================================
# CreativeIngenuity.jl — Creative generation
# ============================================================================
module CreativeIngenuity

export generate

function generate(research, internal)
    Dict(
        :ideas => String[],
        :creative_mode => internal.arousal > 0.6
    )
end

end # module
