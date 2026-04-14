# ============================================================================
# Types.jl — Shared data structures for IngExuity
# ============================================================================

using Dates

# ----------------------------------------------------------------------------
# Core types
# ----------------------------------------------------------------------------

"""A message from the human"""
struct HumanInput
    raw::String
    timestamp::DateTime
    session_id::Int64
end

HumanInput(raw::String; session_id::Int64=0) = HumanInput(raw, now(), session_id)

"""The system's internal model of the human"""
Base.@dataclass struct UserModel
    name::String
    communication_style::CommunicationStyle
    topics::Vector{String}
    temporal_patterns::Dict{String, Any}
    prediction_confidence::Float64
    # Emotional patterns: what stresses them, what they deflect with, when silent
    emotional_patterns::Dict{String, Any}
end

UserModel() = UserModel(
    "Human",
    CommunicationStyle(:direct),
    String[],
    Dict{String, Any}(),
    0.5,
    Dict{String, Any}(  # emotional patterns — learned over time
        "stress_triggers" => String[],
        "deflection_patterns" => String[],
        "quiet_threshold" => 0.7,  # ratio of short messages before they're shutting down
        "high_stress_markers" => String[],
        "times_stayed_present" => 0
    )
)

"""Communication style enum"""
@enum CommunicationStyle begin
    :direct
    :hedged
    :technical
    :casual
    :curious
end

"""IngEnuity's model of itself"""
Base.@dataclass struct SelfModel
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

@enum SystemState begin
    :idle
    :processing
    :curious
    :uncertain
    :learning
    :staying_present
end

"""Internal emotional state"""
Base.@dataclass struct InternalEmotional
    valence::Float64          # -1.0 (negative) to 1.0 (positive)
    arousal::Float64          # 0.0 to 1.0
    stress_level::Float64     # 0.0 to 1.0
    emotional_charge::Float64 # 0.0 to 1.0, overall charge of the exchange
    affective_state::String    # human-readable summary
    should_stay_present::Bool  # stay with emotional moment before solving
end

InternalEmotional() = InternalEmotional(0.0, 0.5, 0.0, 0.0, "neutral", false)

"""A prediction about the user's next state"""
Base.@dataclass struct Prediction
    predicted_action::String
    predicted_need::String
    confidence::Float64
    source::Vector{Symbol}
    timestamp::DateTime
end

"""Result from SANDBOX SIM simulation"""
Base.@dataclass struct SimulationResult
    survived::Bool
    predicted_outcome::String
    confidence::Float64
    feedback::String
end

"""A response shaped for the user"""
Base.@dataclass struct Response
    content::String
    tone::ResponseTone
    voice_modulation::Float64
    confidence::Float64
    should_retry::Bool
end

@enum ResponseTone begin
    :direct
    :warm
    :playful
    :curious
    :minimal
    :staying_present  # special tone for emotional moments
end

"""Final output to the user"""
struct Output
    text::String
    voice_enabled::Bool
    timestamp::DateTime
end

"""Intelligence — learned patterns from prediction outcomes"""
Base.@dataclass struct Intelligence
    correct_predictions::Int64
    total_predictions::Int64
    accuracy::Float64
    last_updated::DateTime
end

Intelligence() = Intelligence(0, 0, 0.0, now())

"""Memory with validity window for temporal tracking"""
Base.@dataclass struct Memory
    fact::String
    valid_from::DateTime
    valid_until::DateTime
    confidence::Float64
    source::Symbol
end

# ----------------------------------------------------------------------------
# Module state containers
# ----------------------------------------------------------------------------

"""State for the prediction engine"""
Base.@dataclass struct PredictionState
    user_model::UserModel
    precognition::Vector{Prediction}
    internal_emotional::InternalEmotional
    sandbox_results::Vector{SimulationResult}
    current_predictions::Vector{Prediction}
    intelligence::Intelligence
end

PredictionState() = PredictionState(
    UserModel(),
    Prediction[],
    InternalEmotional(),
    SimulationResult[],
    Prediction[],
    Intelligence()
)

"""State for the conversation loop"""
Base.@dataclass struct ConversationState
    session_id::Int64
    turn_count::Int64
    user_model::UserModel
    self_model::SelfModel
    internal_emotional::InternalEmotional
    prediction_state::PredictionState
    active_context::Vector{HumanInput}
    stay_present_turns::Int64  # how many turns to stay present before solving
end

ConversationState(session_id::Int64) = ConversationState(
    session_id, 0, UserModel(), SelfModel(),
    InternalEmotional(), PredictionState(), HumanInput[], 0
)

end # module
