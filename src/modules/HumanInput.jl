# ============================================================================
# HumanInput.jl — Input Layer: raw input from the human
# ============================================================================
module HumanInput

using ...Types

"""Process raw human input into a structured HumanInput"""
function process(raw::String; session_id::Int64=0)::HumanInput
    # Basic preprocessing — strip excess whitespace, preserve intentional formatting
    cleaned = strip(replace(raw, r"\s+" => " "))
    HumanInput(cleaned, now(), session_id)
end

"""Check if the input is a clarification request"""
function is_clarification_request(input::HumanInput)::Bool
    # Patterns that indicate the human wants something explained
    raw = lowercase(input.raw)
    clarification_phrases = ["what do you mean", "can you explain", "i don't understand",
                            "clarify", "rephrase", "ELI5", "explain like"]
    any(p -> contains(raw, p), clarification_phrases)
end

"""Check if input is a greeting"""
function is_greeting(input::HumanInput)::Bool
    raw = lowercase(input.raw)
    greetings = ["hello", "hi", "hey", "yo", "sup", "greetings"]
    any(g -> occursin(g, raw), greetings)
end

"""Check if input is a substantive question vs casual"""
function is_substantive(input::HumanInput)::Bool
    raw = input.raw
    length(raw) > 30 || contains(raw, '?')
end

end # module
