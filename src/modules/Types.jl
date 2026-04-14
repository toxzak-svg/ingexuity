# ============================================================================
# Types.jl — Shared data structures for IngExuity v1.3
# ============================================================================

module Types

using Dates

export CommunicationStyle, SystemState, ResponseTone,
       COMMUNICATION_STYLE_DIRECT, COMMUNICATION_STYLE_HEDGED,
       COMMUNICATION_STYLE_TECHNICAL, COMMUNICATION_STYLE_CASUAL,
       COMMUNICATION_STYLE_CURIOUS,
       SYSTEM_STATE_IDLE, SYSTEM_STATE_PROCESSING,
       SYSTEM_STATE_CURIOUS, SYSTEM_STATE_UNCERTAIN,
       SYSTEM_STATE_LEARNING, SYSTEM_STATE_STAYING_PRESENT,
       RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
       RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS,
       RESPONSE_TONE_MINIMAL, RESPONSE_TONE_STAYING_PRESENT,
       HumanInput, UserModel, SelfModel, InternalEmotional,
       Prediction, SimulationResult, Response, Output,
       Intelligence, Memory, PredictionState, ConversationState

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------

@enum CommunicationStyle direct hedged technical casual curious
@enum SystemState idle processing curious uncertain learning staying_present
@enum ResponseTone direct warm playful curious minimal staying_present

const COMMUNICATION_STYLE_DIRECT = CommunicationStyle(0)
const COMMUNICATION_STYLE_HEDGED = CommunicationStyle(1)
const COMMUNICATION_STYLE_TECHNICAL = CommunicationStyle(2)
const COMMUNICATION_STYLE_CASUAL = CommunicationStyle(3)
const COMMUNICATION_STYLE_CURIOUS = CommunicationStyle(4)

const SYSTEM_STATE_IDLE = SystemState(0)
const SYSTEM_STATE_PROCESSING = SystemState(1)
const SYSTEM_STATE_CURIOUS = SystemState(2)
const SYSTEM_STATE_UNCERTAIN = SystemState(3)
const SYSTEM_STATE_LEARNING = SystemState(4)
const SYSTEM_STATE_STAYING_PRESENT = SystemState(5)

const RESPONSE_TONE_DIRECT = ResponseTone(0)
const RESPONSE_TONE_WARM = ResponseTone(1)
const RESPONSE_TONE_PLAYFUL = ResponseTone(2)
const RESPONSE_TONE_CURIOUS = ResponseTone(3)
const RESPONSE_TONE_MINIMAL = ResponseTone(4)
const RESPONSE_TONE_STAYING_PRESENT = ResponseTone(5)

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
        COMMUNICATION_STYLE_DIRECT,
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
    SYSTEM_STATE_IDLE,
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

end # module Types
