# ============================================================================
# IngExuity.jl — Main entry point
# Julia everywhere: runtime + inference + server
# ============================================================================
module IngExuity

using Genie
using Genie.Router
using Genie.Renderer.Html

# Module structure — 16 modules matching IngExuity architecture v1.2
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
include("modules/Action.jl")
include("modules/ReactionObservance.jl")
include("modules/Response.jl")
include("modules/Voice.jl")
include("modules/Output.jl")
include("modules/Understanding.jl")
include("modules/Intelligence.jl")

# ============================================================================
# Conversation loop — the core IngExuity experience
# ============================================================================

const GLOBAL_STATE = Types.ConversationState(0)

"""
    chat(input::String; session_id::Int64=0)::String

The main entry point. Takes a string from the human, runs it through the
IngExuity module pipeline, and returns the response string.
"""
function chat(input::String; session_id::Int64=0)::String
    # === INPUT LAYER ===
    human_input = HumanInput.process(input; session_id=session_id)

    # === RESULTS ANALYSIS (feedback from previous turns) ===
    results = ResultsAnalysis.process(human_input, GLOBAL_STATE)

    # === COGNITIVE PROCESSING ===
    comprehension = Comprehension.comprehend(human_input)

    # Update self model
    self_model = SelfModel.update(GLOBAL_STATE.self_model, human_input, comprehension)

    # Update user model
    user_model = UserModel.update(GLOBAL_STATE.user_model, human_input, comprehension)

    # Update internal emotional state
    internal_emotional = InternalEmotional.update(
        GLOBAL_STATE.internal_emotional,
        human_input,
        comprehension
    )

    # Curiosity check
    curiosity = Curiosity.check(human_input, comprehension, user_model)

    # === RESEARCH & REASONING ===
    research = Research.investigate(human_input, comprehension; curiosity=curiosity)

    decision = Decision.decide(research)

    # Creative Ingenuity (for novel situations)
    creative = CreativeIngenuity.generate(research, internal_emotional)

    # Precognition (long-range trajectory)
    precognition = Precognition.predict_trajectory(user_model, internal_emotional)

    # === PREDICTION ENGINE (PRIMARY) ===
    predictions = Predictions.predict(
        user_model,
        internal_emotional,
        precognition,
        GLOBAL_STATE.prediction_state.sandbox_results;
        context=Dict(:latest_input => input)
    )

    # === SANDBOX SIM ===
    sandbox_results = SandboxSim.simulate_batch(
        predictions,
        user_model,
        self_model
    )

    # Filter to surviving predictions
    surviving_predictions = SandboxSim.filter_surviving(predictions, sandbox_results)

    # === ACTION ===
    action = Action.execute(decision, creative, surviving_predictions)

    # === REACTION OBSERVANCE ===
    reaction = ReactionObservance.observe(human_input, action)

    # === OUTPUT LAYER ===
    # Determine tone
    tone = Voice.determine_tone(internal_emotional, user_model, reaction)

    # Formulate response
    response = Response.formulate(surviving_predictions, comprehension; tone=tone)

    # Adjust for emotional state
    response = Response.adjust_tone(response, internal_emotional)

    # Render output
    output = Output.render(response, comprehension; voice_enabled=true)

    # === UNDERSTANDING + INTELLIGENCE ===
    understanding = Understanding.interpret(
        human_input,
        response,
        surviving_predictions,
        reaction
    )

    # Update intelligence from outcome
    for pred in surviving_predictions
        # Simple heuristic: if we had high confidence, count it as a correct prediction
        was_correct = pred.confidence > 0.7
        Predictions.update_from_outcome!(GLOBAL_STATE.prediction_state, pred, was_correct)
    end

    # === UPDATE GLOBAL STATE ===
    GLOBAL_STATE.turn_count += 1
    GLOBAL_STATE.user_model = user_model
    GLOBAL_STATE.self_model = self_model
    GLOBAL_STATE.internal_emotional = internal_emotional
    push!(GLOBAL_STATE.active_context, human_input)

    output.text
end

"""
    predict_user(;session_id::Int64=0)::Vector{Dict}

Return the current predictions about the user without generating a response.
Useful for debugging and for the dashboard.
"""
function predict_user(;session_id::Int64=0)::Vector{Dict}
    preds = GLOBAL_STATE.prediction_state.current_predictions
    [
        Dict(
            "action" => p.predicted_action,
            "need" => p.predicted_need,
            "confidence" => p.confidence,
            "sources" => [string(s) for s in p.source]
        )
        for p in preds
    ]
end

"""
    get_intelligence()::Dict

Return the current Intelligence metrics.
"""
function get_intelligence()::Dict
    intel = GLOBAL_STATE.prediction_state.intelligence
    Dict(
        "correct_predictions" => intel.correct_predictions,
        "total_predictions" => intel.total_predictions,
        "accuracy" => intel.accuracy,
        "last_updated" => string(intel.last_updated)
    )
end

"""
    get_user_model()::Dict

Return the current User Model.
"""
function get_user_model()::Dict
    um = GLOBAL_STATE.user_model
    Dict(
        "name" => um.name,
        "communication_style" => string(um.communication_style),
        "topics" => um.topics,
        "prediction_confidence" => um.prediction_confidence
    )
end

# ============================================================================
# Genie.jl web server — API + embedded UI
# ============================================================================

route("/") do
    html(gensub_view())
end

route("/api/chat", method=POST) do
    params = Genie.Router.jsonpayload()
    input = get(params, "message", "")
    isempty(input) && return Genie.Renderer.json(Dict("error" => "no message"))
    response = chat(input)
    Genie.Renderer.json(Dict("response" => response))
end

route("/api/predict") do
    Genie.Renderer.json(Dict("predictions" => predict_user()))
end

route("/api/intelligence") do
    Genie.Renderer.json(get_intelligence())
end

route("/api/user_model") do
    Genie.Renderer.json(get_user_model())
end

route("/health") do
    "ok"
end

# ============================================================================
# Start the server
# ============================================================================

"""
    start(host::String="0.0.0.0", port::Int=8000)

Start the IngExuity web server.
"""
function start(host::String="0.0.0.0", port::Int=8000)
    Genie.config.server_host = host
    Genie.config.server_port = port
    up(host, port)
    println("IngExuity running at http://$host:$port")
end

end # module
