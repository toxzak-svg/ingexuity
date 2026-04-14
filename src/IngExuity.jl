# ============================================================================
# IngExuity.jl — Main entry point (minimal HTTP server, no external deps)
# Julia everywhere: runtime + inference + server
# ============================================================================
module IngExuity

using HTTP, Dates, Base.Iterators

# Module structure — 16 modules + Memory matching IngEnuity architecture v1.3
include("modules/Types.jl")
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
include("modules/Predictions.jl")
include("modules/SandboxSim.jl")
include("modules/Memory.jl")
include("modules/Action.jl")
include("modules/ReactionObservance.jl")
include("modules/Response.jl")
include("modules/Voice.jl")
include("modules/Output.jl")
include("modules/Understanding.jl")
include("modules/Intelligence.jl")

# ============================================================================
# Conversation loop — the core IngEnuity experience
# ============================================================================

const GLOBAL_STATE = Types.ConversationState(0)

function chat(input::String; session_id::Int64=0)::String
    human_input = HumanInput.process(input; session_id=session_id)
    results = ResultsAnalysis.process(human_input, GLOBAL_STATE)
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
    decision = Decision.decide(research)
    creative = CreativeIngenuity.generate(research, internal_emotional)
    precognition = Precognition.predict_trajectory(user_model, internal_emotional)

    predictions = Predictions.predict(
        user_model, internal_emotional, precognition,
        GLOBAL_STATE.prediction_state.sandbox_results
    )

    sandbox_results = SandboxSim.simulate_batch(predictions, user_model, self_model)
    surviving_predictions = SandboxSim.filter_surviving(predictions, sandbox_results)
    action = Action.execute(decision, creative, surviving_predictions)
    reaction = ReactionObservance.observe(human_input, action)
    tone = Voice.determine_tone(internal_emotional, user_model, reaction)
    response = Response.formulate(surviving_predictions, comprehension; tone=tone)
    response = Response.adjust_tone(response, internal_emotional)
    output = Output.render(response, comprehension; voice_enabled=true)
    understanding = Understanding.interpret(human_input, response, surviving_predictions, reaction)

    Memory.store("User said: $(human_input.raw)", source=:conversation)
    if !isempty(comprehension[:topic]) && comprehension[:topic] != "general"
        Memory.store("Topic: $(comprehension[:topic])", source=:topic_detection)
    end

    for pred in surviving_predictions
        was_correct = pred.confidence > 0.7
        Predictions.update_from_outcome!(GLOBAL_STATE.prediction_state, pred, was_correct)
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

# ============================================================================
# Minimal HTTP server
# ============================================================================

function handle_request(req::HTTP.Request)::HTTP.Response
    target = req.target

    if startswith(target, "/health")
        return HTTP.Response(200, ["Content-Type" => "text/plain"], "ok")
    elseif target == "/api/chat" && HTTP.method(req) == "POST"
        body = String(req.body)
        m = match(r"\"message\"\s*:\s*\"([^\"]+)\"", body)
        input = m !== nothing ? m[1] : ""
        isempty(input) && return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(Dict("error" => "no message")))
        response = chat(input)
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(Dict("response" => response)))
    elseif target == "/api/predict"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(Dict("predictions" => predict_user())))
    elseif target == "/api/intelligence"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_intelligence()))
    elseif target == "/api/user_model"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_user_model()))
    elseif target == "/api/memory"
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=to_json(get_memory_summary()))
    elseif target == "/"
        return HTTP.Response(200, ["Content-Type" => "text/html"], body=HTML_UI)
    else
        return HTTP.Response(404, "Not found")
    end
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
</style>
</head>
<body>
<h1>IngExuity</h1>
<div id="chat"></div>
<input id="input" placeholder="Say something..." onkeydown="if(event.keyCode===13)send()">
<button onclick="send()" style="margin-top:10px">Send</button>
<div id="stats"></div>
<script>
async function send() {
  const inp = document.getElementById('input'), chat = document.getElementById('chat');
  const text = inp.value.trim(); if(!text) return;
  inp.value = '';
  const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: text})});
  const data = await res.json();
  chat.innerHTML += '<div class="msg user">' + text + '</div>';
  chat.innerHTML += '<div class="msg ai">' + data.response + '</div>';
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

start() = HTTP.serve(handle_request, "0.0.0.0", server_port())

end # module
