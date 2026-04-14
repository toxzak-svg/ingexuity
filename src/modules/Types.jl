# ============================================================================
# Types.jl — Shared data structures for IngExuity v1.3
# ============================================================================

using Dates

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------

@enum CommunicationStyle :direct :hedged :technical :casual :curious
@enum SystemState :idle :processing :curious :uncertain :learning :staying_present
@enum ResponseTone :direct :warm :playful :curious :minimal :staying_present

# ----------------------------------------------------------------------------
# Core types
# ----------------------------------------------------------------------------

"""A message from the human"""
struct HumanInput
    raw::String
    timestamp::Dates.DateTime
    session_id::Int64
end
HumanInput(raw::String; session_id::Int64=0) = HumanInput(raw, Dates.now(), session_id)

"""The system's internal model of the human"""
mutable struct UserModel
    name::String
    communication_style::CommunicationStyle
    topics::Vector{String}
    temporal_patterns::Dict{String,Any}
    prediction_confidence::Float64
    emotional_patterns::Dict{String,Any}
end
function UserModel()
    UserModel(
        "Human",
        CommunicationStyle(:direct),
        String[],
        Dict{String,Any}(),
        0.5,
        Dict{String,Any}(
            "stress_triggers" => String[],
            "deflection_patterns" => String[],
            "quiet_threshold" => 0.7,
            "high_stress_markers" => String[],
            "times_stayed_present" => 0,
            "quiet_count" => 0,
            "is_quiet" => false
        )
    )
end

"""IngEnuity's model of itself"""
mutable struct SelfModel
    identity::String
    capabilities::Vector{String}
    limitations::Vector{String}
    current_state::SystemState
    confidence::Float64
end
SelfModel() = SelfModel(
    "IngEnuity",
    ["reasoning", "prediction", "conversation", "learning"],
    ["perfect memory", "infinite context"],
    SystemState(:idle),
    0.9
)

"""Internal emotional state"""
mutable struct InternalEmotional
    valence::Float64
    arousal::Float64
    stress_level::Float64
    emotional_charge::Float64
    affective_state::String
    should_stay_present::Bool
end
InternalEmotional() = InternalEmotional(0.0, 0.5, 0.0, 0.0, "neutral", false)

"""A prediction about the user's next state"""
struct Prediction
    predicted_action::String
    predicted_need::String
    confidence::Float64
    source::Vector{Symbol}
    timestamp::Dates.DateTime
end

"""Result from SANDBOX SIM simulation"""
struct SimulationResult
    survived::Bool
    predicted_outcome::String
    confidence::Float64
    feedback::String
end

"""A response shaped for the user"""
struct Response
    content::String
    tone::ResponseTone
    voice_modulation::Float64
    confidence::Float64
    should_retry::Bool
end

"""Final output to the user"""
struct Output
    text::String
    voice_enabled::Bool
    timestamp::Dates.DateTime
end

"""Intelligence — learned patterns from prediction outcomes"""
mutable struct Intelligence
    correct_predictions::Int64
    total_predictions::Int64
    accuracy::Float64
    last_updated::Dates.DateTime
end
Intelligence() = Intelligence(0, 0, 0.0, Dates.now())

"""Memory with validity window for temporal tracking"""
struct Memory
    fact::String
    valid_from::Dates.DateTime
    valid_until::Dates.DateTime
    confidence::Float64
    source::Symbol
end

"""State for the prediction engine"""
mutable struct PredictionState
    user_model::UserModel
    precognition::Vector{Prediction}
    internal_emotional::InternalEmotional
    sandbox_results::Vector{SimulationResult}
    current_predictions::Vector{Prediction}
    intelligence::Intelligence
end
PredictionState() = PredictionState(
    UserModel(), Prediction[], InternalEmotional(),
    SimulationResult[], Prediction[], Intelligence()
)

"""State for the conversation loop"""
mutable struct ConversationState
    session_id::Int64
    turn_count::Int64
    user_model::UserModel
    self_model::SelfModel
    internal_emotional::InternalEmotional
    prediction_state::PredictionState
    active_context::Vector{HumanInput}
    stay_present_turns::Int64
end
ConversationState(session_id::Int64) = ConversationState(
    session_id, 0, UserModel(), SelfModel(),
    InternalEmotional(), PredictionState(), HumanInput[], 0
)
