# ============================================================================
# Decision.jl — Decision making
# ============================================================================
module Decision

export decide

function decide(research)
    Dict(
        :action => "respond",
        :confidence => 0.8,
        :reasoning => "respond normally"
    )
end

end # module
