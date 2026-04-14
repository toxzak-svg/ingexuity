# ============================================================================
# Understanding.jl — Interpret and learn from conversations
# ============================================================================
module Understanding

export interpret

function interpret(human_input, response, predictions, reaction)
    Dict(
        :understood => true,
        :learning => length(predictions) > 0,
        :confidence => length(predictions) > 0 ? mean([p.confidence for p in predictions]) : 0.5
    )
end

end # module
