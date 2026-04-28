# ============================================================================
# Output.jl — Render the final output
# v2: Multi-channel output (text, voice, actions)
# ============================================================================
module Output

using ..Types: Output as OutputType, Response as ResponseType
using Dates

export render, render_multi_channel, add_action

function render(response::ResponseType, comprehension; voice_enabled::Bool=false)
    OutputType(response.content, voice_enabled, Dates.now())
end

function render_multi_channel(response::ResponseType, comprehension;
                              voice_enabled::Bool=false,
                              empathy_level::Float64=0.5)::Dict{Symbol,Any}
    text = response.content
    voice = voice_enabled ? text : ""

    actions = []

    modulation = response.voice_modulation

    if empathy_level > 0.7 && !occursin("?", text)
        push!(actions, Dict(:type => "validate_emotion", :priority => "high"))
    end

    if occursin("but", lowercase(text)) || occursin("however", lowercase(text))
        push!(actions, Dict(:type => "acknowledge_tension", :priority => "medium"))
    end

    Dict{Symbol,Any}(
        :text => text,
        :voice => voice,
        :actions => actions,
        :modulation => modulation,
        :empathy_level => empathy_level,
        :timestamp => Dates.now()
    )
end

function add_action(output::Dict{Symbol,Any}, action_type::String, priority::String="medium")
    actions = get(output, :actions, [])
    push!(actions, Dict(:type => action_type, :priority => priority))
    output[:actions] = actions
    output
end

end # module
