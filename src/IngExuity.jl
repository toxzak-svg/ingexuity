# ============================================================================
# IngExuity.jl — Main entry point (minimal HTTP server, no external deps)
# Julia everywhere: runtime + inference + server
# ============================================================================
module IngExuity

using HTTP, Dates, Base.Iterators, JSON

# Module structure — 20 modules + Memory matching IngEnuity architecture v1.4
include("modules/Types.jl")

# Main chat function (local inference + module pipeline)
export chat, chat_template, predict_user, get_intelligence, get_user_model, get_memory_summary
export load_local_model, load_local_tokenizer, is_local_loaded
export is_live_data_query, fetch_live_data_for_message
export start
include("modules/HumanInput.jl")
include("modules/ResultsAnalysis.jl")
include("modules/Comprehension.jl")
include("modules/SelfModel.jl")
include("modules/UserModel.jl")
include("modules/InternalEmotional.jl")
include("modules/Curiosity.jl")
include("modules/Research.jl")
include("modules/CreativeIngenuity.jl")
include("modules/Decision.jl")
include("modules/Precognition.jl")
include("modules/SandboxSim.jl")
include("modules/Predictions.jl")
include("modules/Memory.jl")
include("modules/Action.jl")
include("modules/ReactionObservance.jl")
include("modules/Response.jl")
include("modules/Voice.jl")
include("modules/Output.jl")
include("modules/Understanding.jl")
include("modules/Intelligence.jl")
include("modules/LlamaInference.jl")
# v1: TrainedModel.jl (LoRA) and NanoGPT.jl/IGSDCore.jl (pure Julia) kept in tree
# but disabled until validated. Single backend = LlamaCpp GGUF.
# include("modules/TrainedModel.jl")
# include("modules/NanoGPT.jl")
# include("modules/IGSDCore.jl")

# Bring live-data helpers into scope (defined in Research module)
using .Research: is_live_data_query, fetch_live_data_for_message

# ============================================================================
# Conversation loop — the core IngEnuity experience
# ============================================================================

const GLOBAL_STATE = Types.ConversationState(0)

function chat_template(input::String; session_id::Int64=0)::String
    human_input = HumanInput.process(input; session_id=session_id)
    results = ResultsAnalysis.analyze_turn(human_input, GLOBAL_STATE)
    comprehension = Comprehension.comprehend(human_input)
    self_model = SelfModel.update(GLOBAL_STATE.self_model, human_input, comprehension)
    user_model = UserModel.update(GLOBAL_STATE.user_model, human_input, comprehension)
    internal_emotional = InternalEmotional.update(GLOBAL_STATE.internal_emotional, human_input, comprehension)

    # === PRESENCING CHECK ===
    if InternalEmotional.should_stay_present(internal_emotional)
        stay_response = build_stay_present_response(internal_emotional, user_model)
        new_state, updated_model = InternalEmotional.advance_stay(internal_emotional, user_model)
        GLOBAL_STATE.internal_emotional = new_state
        GLOBAL_STATE.user_model = updated_model
        GLOBAL_STATE.turn_count += 1
        return stay_response
    end

    curiosity = Curiosity.check(human_input, comprehension, user_model)
    research = Research.investigate(human_input, comprehension; curiosity=curiosity)

    live_context = nothing
    if get(research, :requires_live_data, false) && research[:live_data] !== nothing
        live_context = research[:live_data]
    end

    decision = Decision.decide(research)
    creative = CreativeIngenuity.generate(research, internal_emotional)
    precognition = Precognition.predict_trajectory(user_model, internal_emotional)

    predictions = Predictions.predict_with_retry(
        user_model, internal_emotional, precognition,
        GLOBAL_STATE.prediction_state.sandbox_results
    )

    sandbox_results = SandboxSim.validate_batch(predictions, user_model, internal_emotional)
    surviving_predictions = SandboxSim.filter_surviving(predictions, sandbox_results)

    GLOBAL_STATE.prediction_state.current_predictions = surviving_predictions
    GLOBAL_STATE.prediction_state.sandbox_results = sandbox_results

    action = Action.execute(decision, creative, surviving_predictions)
    reaction = ReactionObservance.observe(human_input, action)
    tone = Voice.determine_tone(internal_emotional, user_model, reaction)  # Symbol now
    response = Response.formulate(surviving_predictions, comprehension; tone=tone, live_context=live_context)  # Symbol
    response = Response.adjust_tone(response, internal_emotional)
    output = Output.render(response, comprehension; voice_enabled=true)
    understanding = Understanding.interpret(human_input, response, surviving_predictions, reaction)

    Memory.store("User said: $(human_input.raw)", source=:conversation)
    topic = comprehension[:topic]
    if !isempty(string(topic)) && topic != :general
        Memory.store("Topic: $(string(topic))", source=:topic_detection)
    end

    for pred in surviving_predictions
        was_correct = ReactionObservance.evaluate_prediction_accuracy(pred, reaction)
        Predictions.update_from_outcome!(GLOBAL_STATE.prediction_state, pred, was_correct)
    end

    if reaction[:emotional_shift] !== :neutral
        Precognition.update_trajectories!(user_model, internal_emotional, "emotional_shift",
            Dict("direction" => string(reaction[:emotional_shift])))
    end

    GLOBAL_STATE.turn_count += 1
    GLOBAL_STATE.user_model = user_model
    GLOBAL_STATE.self_model = self_model
    GLOBAL_STATE.internal_emotional = internal_emotional
    push!(GLOBAL_STATE.active_context, human_input)

    output.text
end

function build_stay_present_response(internal::Types.InternalEmotional, user_model::Types.UserModel)::String
    if internal.stress_level > 0.7
        "That sounds really hard. I'm here — take your time."
    elseif internal.valence < -0.4
        "That sounds discouraging. What matters most to you right now?"
    elseif internal.emotional_charge > 0.8
        "There's a lot in that. I'm listening — what do you want to focus on?"
    elseif user_model.emotional_patterns["is_quiet"]
        "You seem quiet. You don't have to say anything — I'm here."
    else
        "That sounds meaningful. I'm here — go on."
    end
end

# ============================================================================
# LOCAL INFERENCE — LlamaCpp GGUF (single backend for v1)
# No external API. Runs fully offline on device.
# NanoGPT/TrainedModel/IGSDCore are in the tree but disabled until validated.
# ============================================================================

function load_local_model()
    @info "Loading LlamaCpp GGUF model..."
    return LlamaInference.load_llama_model()
end

function is_local_loaded()::Bool
    LlamaInference.is_loaded()
end

# Kept as no-ops for API compatibility; v1 uses GGUF via LlamaInference.
load_local_tokenizer() = nothing

function chat(input::String; session_id::Int64=0)::Dict{String,Any}
    live_data = nothing
    if is_live_data_query(input)
        try
            live_data = fetch_live_data_for_message(input)
        catch e
            @debug "Live data fetch failed" exception=e
        end
    end

    prompt = build_prompt_with_context(input, live_data)

    try
        if LlamaInference.is_loaded()
            result = LlamaInference.chat(prompt; session_id=session_id)
            if haskey(result, "text") && !isempty(result["text"]) && !haskey(result, "error")
                result["model"] = "Llama-3.2-1B-Instruct (GGUF)"
                result["live_data_used"] = live_data !== nothing
                track_basic_prediction(input, result)
                return result
            end
        end
    catch e
        @debug "LlamaInference not available" exception=e
    end

    try
        text = chat_template(input; session_id=session_id)
        return Dict("text" => text, "model" => "IngExuity-Module-Pipeline", "session_id" => session_id)
    catch e
        return Dict("error" => string(e), "text" => "", "session_id" => session_id)
    end
end

function build_prompt_with_context(input::String, live_data::Union{Dict{String,Any},Nothing})::String
    if live_data === nothing
        return input
    end

    results = get(live_data, "results", [])
    if isempty(results)
        return input
    end

    top = results[1]
    title = get(top, "title", "")
    snippet = get(top, "snippet", "")
    query_type = get(live_data, "query_type", "general")

    context_prefix = if query_type == "weather"
        "[Current weather: $snippet] "
    elseif query_type == "stock"
        "[Stock info: $snippet] "
    elseif query_type == "news"
        "[Latest news: $snippet] "
    elseif query_type == "factual"
        "[Info: $snippet] "
    else
        "[Current info: $snippet] "
    end

    return context_prefix * input
end

# Track basic prediction from direct LLM responses
function track_basic_prediction(input::String, result::Dict{String,Any})
    try
        text = get(result, "text", "")
        if isempty(text) return end
        
        # Initialize prediction_state if needed
        if !isdefined(GLOBAL_STATE, :prediction_state) || GLOBAL_STATE.prediction_state === nothing
            GLOBAL_STATE.prediction_state = Types.PredictionState()
        end
        
        # Basic prediction: user needs info/support (positional constructor)
        pred = Types.Prediction(
            "continue_conversation",
            "supportive_response",
            0.6,
            Symbol[:direct_llm],
            Dates.now()
        )
        
        # Simple reaction: assume positive if response is non-empty
        reaction = Dict{Symbol,Any}(:emotional_shift => :neutral, :engagement => 1.0)
        
        # Update prediction stats
        Predictions.update_from_outcome!(GLOBAL_STATE.prediction_state, pred, true)
        
        # Store in memory
        Memory.store("User: $input", source=:conversation)
        Memory.store("IngExuity: $text", source=:response)
        
        # Debug: log updated intelligence
        intel = GLOBAL_STATE.prediction_state.intelligence
        @info "Prediction tracked: correct=$(intel.correct_predictions), total=$(intel.total_predictions)"
    catch e
        @warn "Prediction tracking failed" exception=e
    end
end

function predict_user()::Vector{Dict}
    [Dict("action" => p.predicted_action, "need" => p.predicted_need,
          "confidence" => p.confidence, "sources" => [string(s) for s in p.source])
     for p in GLOBAL_STATE.prediction_state.current_predictions]
end

function get_intelligence()::Dict
    i = GLOBAL_STATE.prediction_state.intelligence
    Dict("correct" => i.correct_predictions, "total" => i.total_predictions,
         "accuracy" => i.accuracy, "last_updated" => string(i.last_updated))
end

function get_user_model()::Dict
    um = GLOBAL_STATE.user_model
    Dict("name" => um.name, "communication_style" => string(um.communication_style),
         "topics" => um.topics, "prediction_confidence" => um.prediction_confidence,
         "is_stressed" => UserModel.is_stressed(um))
end

function get_memory_summary()::Dict
    Dict("facts_stored" => Memory.count(), "valid_facts" => length(Memory.retrieve()))
end

# ============================================================================
# Simple JSON helpers (no external deps)
# ============================================================================

json_value(v::Nothing) = "null"
json_value(v::Bool) = v ? "true" : "false"
json_value(v::Real) = string(v)
json_value(v::AbstractString) = "\"$(replace(replace(v, "\\" => "\\\\"), "\"" => "\\\""))\"" 
json_value(v::Vector{UInt8}) = string(v)

function json_value(v::Vector)
    items = join([json_value(x) for x in v], ",")
    "[$items]"
end

function json_value(v::Dict)
    pairs = ["\"$k\": $(json_value(val))" for (k, val) in v]
    "{$(join(pairs, ","))}"
end

to_json(d::Dict) = json_value(d)

function start_llama_model()
    try
        model_path = joinpath(@__DIR__, "..", "models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        if isfile(model_path)
            @info "Pre-loading Llama 3.2 model from: $model_path"
            LlamaInference.load_llama_model(model_path=model_path)
            # Warm the model into llama.cpp's memory before the HTTP listener
            # accepts traffic. Without this, the first /api/chat request pays
            # a 5-15s cold-load penalty.
            LlamaInference.warmup()
            @info "Llama 3.2 model ready"
        else
            @warn "Llama 3.2 GGUF not found at: $model_path"
        end
    catch e
        @warn "Could not pre-load Llama 3.2: $e"
    end
end

# ============================================================================
# Minimal HTTP server
# ============================================================================

function handle_request(req::HTTP.Request)::HTTP.Response
    target = req.target

    if startswith(target, "/health")
        return HTTP.Response(200, ["Content-Type" => "text/plain"], "ok")
    elseif target == "/api/chat" && HTTP.method(req) == "POST"
        return handle_chat(req)
    elseif target == "/api/predict"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(Dict("predictions" => predict_user())))
    elseif target == "/api/intelligence"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_intelligence()))
    elseif target == "/api/user_model"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_user_model()))
    elseif target == "/api/memory"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_memory_summary()))
    elseif target == "/api/local/status" && HTTP.method(req) == "GET"
        info = LlamaInference.get_model_info()
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(info))
    elseif target == "/api/local/load" && HTTP.method(req) == "POST"
        try
            model_path = LlamaInference.load_llama_model()
            return HTTP.Response(200, ["Content-Type" => "application/json"],
                body=to_json(Dict("loaded" => true, "model" => "Llama-3.2-1B-Instruct")))
        catch e
            return HTTP.Response(500, ["Content-Type" => "application/json"],
                body=to_json(Dict("error" => string(e))))
        end

    elseif target == "/"
        return HTTP.Response(200, ["Content-Type" => "text/html"], body=HTML_UI)
    else
        return HTTP.Response(404, "Not found")
    end
end

function handle_chat(req::HTTP.Request)::HTTP.Response
    input = ""
    try
        body = String(req.body)
        isempty(body) || (input = get(JSON.parse(body), "message", "") |> string)
    catch e
        return HTTP.Response(400, ["Content-Type" => "application/json"],
            body=to_json(Dict("error" => "invalid JSON body: $(e)")))
    end
    if isempty(input)
        return HTTP.Response(400, ["Content-Type" => "application/json"],
            body=to_json(Dict("error" => "missing 'message' field")))
    end
    chat_response = chat(input)
    return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(chat_response))
end

const HTML_UI = """
<!DOCTYPE html>
<html>
<head>
<title>IngExuity</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #0a0a0a; color: #e0e0e0; }
h1 { color: #7aff7a; font-size: 1.5em; }
#chat { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
.msg { padding: 10px 14px; border-radius: 12px; max-width: 80%; }
.msg.user { align-self: flex-end; background: #1a3a1a; }
.msg.ai { align-self: flex-start; background: #1a1a2a; }
input { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #111; color: #fff; box-sizing: border-box; }
button { padding: 12px 24px; background: #7aff7a; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
#stats { font-size: 0.85em; color: #888; margin-top: 20px; }
#status { font-size: 0.75em; color: #666; margin-top: 10px; }
.error { color: #ff6b6b; }
</style>
</head>
<body>
<h1>IngExuity</h1>
<div id="chat"></div>
<input id="input" placeholder="Say something..." onkeydown="if(event.keyCode===13)send()">
<button onclick="send()" style="margin-top:10px">Send</button>
<div id="stats"></div>
<div id="status"></div>
<script>
async function send() {
  const inp = document.getElementById('input'), chat = document.getElementById('chat');
  const text = inp.value.trim(); if(!text) return;
  inp.value = '';
  const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: text})});
  const data = await res.json();
  chat.innerHTML += '<div class="msg user">' + text + '</div>';
  if (data.error) {
    chat.innerHTML += '<div class="msg ai error">Error: ' + data.error + '</div>';
  } else {
    chat.innerHTML += '<div class="msg ai">' + (data.text || data.response || 'No response') + '</div>';
    if (data.actions && data.actions.length > 0) {
      console.log('Actions:', data.actions);
    }
  }
  chat.scrollTop = chat.scrollHeight;
  updateStats();
}
async function updateStats() {
  const r = await fetch('/api/intelligence');
  const d = await r.json();
  document.getElementById('stats').innerText = 'Intelligence: ' + d.correct + '/' + d.total + ' predictions';
}
updateStats();
</script>
</body>
</html>
"""

function server_port()
    port_str = get(ENV, "PORT", "8000")
    try
        parse(Int, port_str)
    catch
        8000
    end
end

start() = (start_llama_model(); HTTP.serve(handle_request, "0.0.0.0", server_port()))

end # module
