# ============================================================================
# GemmaProvider.jl — DEPRECATED (2026-04-28)
#
# This module is kept for reference only. IngExuity now uses a Julia-native
# transformer (Flux.jl, see SPEC.md). No external API dependency.
#
# LLM provider for Gemma 4 E2B via Python HTTP service
# Integrates native function calling and audio into IngExuity
# ============================================================================
module GemmaProvider
@warn "GemmaProvider is deprecated. IngExuity now uses a Julia-native transformer built with Flux.jl. See SPEC.md."

using HTTP, JSON

export GemmaLLM, generate, generate_audio, load_model, unload_model,
       is_loaded, get_capabilities

const DEFAULT_HOST = "localhost"
const DEFAULT_PORT = 8765

mutable struct GemmaLLM
    host::String
    port::Int
    loaded::Bool
    capabilities::Dict{Symbol,Any}
end

GemmaLLM(; host=DEFAULT_HOST, port=DEFAULT_PORT) = GemmaLLM(host, port, false, Dict{Symbol,Any}())

function base_url(llm::GemmaLLM)::String
    "http://$(llm.host):$(llm.port)"
end

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

function load_model(llm::GemmaLLM)::Bool
    try
        response = HTTP.post("$(base_url(llm))/load")
        if response.status == 200
            llm.loaded = true
            update_capabilities!(llm)
            return true
        end
    catch e
        @warn "Failed to load Gemma model" exception=e
    end
    false
end

function unload_model(llm::GemmaLLM)::Bool
    try
        response = HTTP.post("$(base_url(llm))/unload")
        if response.status == 200
            llm.loaded = false
            return true
        end
    catch e
        @warn "Failed to unload Gemma model" exception=e
    end
    false
end

function is_loaded(llm::GemmaLLM)::Bool
    llm.loaded
end

function update_capabilities!(llm::GemmaLLM)
    try
        response = HTTP.get("$(base_url(llm))/capabilities")
        if response.status == 200
            data = JSON.parse(String(response.body))
            llm.capabilities = Dict(Symbol(k) => v for (k, v) in data)
        end
    catch e
        @warn "Failed to get capabilities" exception=e
    end
end

function get_capabilities(llm::GemmaLLM)::Dict{Symbol,Any}
    llm.capabilities
end

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

"""
    generate(llm, messages; max_tokens=512, thinking=false)

Generate a response from Gemma.

messages = [
    Dict("role" => "system", "content" => "You are helpful."),
    Dict("role" => "user", "content" => "Hello!")
]
"""
function generate(llm::GemmaLLM, messages::Vector{Dict{String,Any}};
                 max_tokens::Int=512, thinking::Bool=false)::Dict{String,Any}
    payload = Dict(
        "messages" => messages,
        "max_tokens" => max_tokens,
        "enable_thinking" => thinking
    )

    try
        response = HTTP.post(
            "$(base_url(llm))/generate",
            ["Content-Type" => "application/json"],
            JSON.json(payload)
        )

        if response.status == 200
            return JSON.parse(String(response.body))
        else
            return Dict("error" => "HTTP $(response.status)", "text" => "", "actions" => [])
        end
    catch e
        return Dict("error" => string(e), "text" => "", "actions" => [])
    end
end

"""
    generate(llm, system_prompt, user_input; kwargs...)

Simple string-based generation.
"""
function generate(llm::GemmaLLM, system_prompt::String, user_input::String; kwargs...)::Dict{String,Any}
    messages = [
        Dict("role" => "system", "content" => system_prompt),
        Dict("role" => "user", "content" => user_input)
    ]
    generate(llm, messages; kwargs...)
end

# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

"""
    generate_audio(llm, audio_bytes, messages; kwargs...)

Send audio input and get audio + text response.
"""
function generate_audio(llm::GemmaLLM, audio_bytes::Vector{UInt8},
                        messages::Vector{Dict{String,Any}};
                        max_tokens::Int=512)::Dict{String,Any}
    audio_b64 = Base64.base64encode(audio_bytes)
    payload = Dict(
        "audio" => audio_b64,
        "messages" => messages,
        "max_tokens" => max_tokens
    )

    try
        response = HTTP.post(
            "$(base_url(llm))/generate_audio",
            ["Content-Type" => "application/json"],
            JSON.json(payload)
        )

        if response.status == 200
            return JSON.parse(String(response.body))
        else
            return Dict("error" => "HTTP $(response.status)", "text" => "", "actions" => [])
        end
    catch e
        return Dict("error" => string(e), "text" => "", "actions" => [])
    end
end

"""
    generate_audio(llm, audio_path, user_text; kwargs...)

Simple audio file + text input.
"""
function generate_audio(llm::GemmaLLM, audio_path::String, user_text::String; kwargs...)::Dict{String,Any}
    audio_bytes = read(audio_path)
    messages = [Dict("role" => "user", "content" => user_text)]
    generate_audio(llm, audio_bytes, messages; kwargs...)
end

# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

"""
    execute_actions(actions::Vector{Dict}, context::Dict)

Execute function calls returned by Gemma.
This is where system integration happens.
"""
function execute_actions(actions::Vector{Dict}, context::Dict=Dict())::Vector{Dict}
    results = []

    for action in actions
        func = get(action, "function", "")
        params = get(action, "parameters", Dict())

        result = if func == "speak" || func == "tts" || func == "say"
            execute_speak(get(params, "text", ""))
        elseif func == "check_memory" || func == "retrieve"
            execute_memory_lookup(get(params, "query", ""))
        elseif func == "store_memory" || func == "remember"
            execute_memory_store(get(params, "fact", ""), context)
        elseif func == "predict" || func == "anticipate"
            execute_prediction(get(params, "about", ""), context)
        elseif func == "check_time" || func == "time"
            Dict("type" => "text", "content" => string(now()))
        elseif func == "log" || func == "debug"
            @info "Gemma action: $func" params...
            Dict("type" => "ok", "content" => "logged")
        else
            Dict("type" => "error", "content" => "Unknown function: $func")
        end

        push!(results, Dict(
            "function" => func,
            "result" => result
        ))
    end

    results
end

function execute_speak(text::String)::Dict{String,Any}
    Dict("type" => "speech", "text" => text)
end

function execute_memory_lookup(query::String)::Dict{String,Any}
    # Hook into IngExuity's Memory module
    try
        results = Main.Memory.search(query)
        if !isempty(results)
            return Dict("type" => "memory", "content" => [r.fact for r in results])
        end
    catch e
        # Memory module not loaded yet
    end
    Dict("type" => "memory", "content" => [])
end

function execute_memory_store(fact::String, context::Dict)::Dict{String,Any}
    try
        Main.Memory.store(fact, source=:gemma)
        return Dict("type" => "ok", "content" => "stored")
    catch e
        return Dict("type" => "error", "content" => "Failed to store")
    end
end

function execute_prediction(about::String, context::Dict)::Dict{String,Any}
    # Hook into IngExuity's Predictions module
    Dict("type" => "prediction", "content" => "prediction logic here")
end

# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

"""
    chat(llm, conversation_history, user_input; system_prompt=default_prompt)

Full conversation loop with Gemma.
"""
function chat(llm::GemmaLLM, history::Vector{Dict{String,Any}},
              user_input::String;
              system_prompt::String="You are IngExuity, a helpful AI companion.",
              thinking::Bool=false,
              execute::Bool=true)::Dict{String,Any}
    messages = vcat([
        Dict("role" => "system", "content" => system_prompt)
    ], history, [
        Dict("role" => "user", "content" => user_input)
    ])

    response = generate(llm, messages; thinking=thinking)

    if execute && haskey(response, "actions") && !isempty(response["actions"])
        action_results = execute_actions(response["actions"], Dict(
            "history" => history,
            "user_input" => user_input
        ))
        response["action_results"] = action_results
    end

    response
end

end # module