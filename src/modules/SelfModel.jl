# ============================================================================
# SelfModel.jl — IngExuity's self-awareness
# v3: Tracks capabilities, limitations, confidence states, and self-assessment
# Knows when it's uncertain, when it's confident, feeds into Internal/Emotional
# ============================================================================
module SelfModel

using ..Types: SelfModel as SelfModelType, SYSTEM_STATE_IDLE, SYSTEM_STATE_PROCESSING,
               SYSTEM_STATE_CURIOUS, SYSTEM_STATE_UNCERTAIN, SYSTEM_STATE_LEARNING,
               SYSTEM_STATE_STAYING_PRESENT, SystemState

export update, assess_capability, get_self_description, update_confidence!,
       know_self_limits, get_uncertainty_level, get_capability_confidence

const CAPABILITY_TAGS = [:reasoning, :prediction, :conversation, :learning, :empathy,
                          :creativity, :memory, :staying_present, :directness]

const CAPABILITY_CONFIDENCE = Dict{Symbol,Float64}(
    :reasoning => 0.9,
    :prediction => 0.7,
    :conversation => 0.85,
    :learning => 0.75,
    :empathy => 0.8,
    :creativity => 0.65,
    :memory => 0.95,
    :staying_present => 0.85,
    :directness => 0.7
)

function update(self_model::SelfModelType, human_input, comprehension;
                prediction_accuracy::Float64=0.5,
                recent_failures::Int=0)::SelfModelType
    is_question = get(comprehension, :is_question, false)
    topic = get(comprehension, :topic, :general)

    self_model.current_state = determine_state(is_question, recent_failures,
                                               prediction_accuracy, self_model.confidence)

    self_model.confidence = compute_self_confidence(prediction_accuracy,
                                                    recent_failures,
                                                    self_model.current_state)

    update_capabilities!(self_model, topic, comprehension)

    self_model
end

function determine_state(is_question::Bool, recent_failures::Int,
                         prediction_accuracy::Float64, current_confidence::Float64)::SystemState
    if recent_failures >= 3
        return SYSTEM_STATE_UNCERTAIN
    elseif is_question
        return SYSTEM_STATE_CURIOUS
    elseif prediction_accuracy < 0.4 && current_confidence < 0.6
        return SYSTEM_STATE_LEARNING
    elseif current_confidence > 0.8
        return SYSTEM_STATE_PROCESSING
    elseif current_confidence < 0.4
        return SYSTEM_STATE_UNCERTAIN
    else
        return SYSTEM_STATE_IDLE
    end
end

function compute_self_confidence(prediction_accuracy::Float64, recent_failures::Int,
                                 current_state::SystemState)::Float64
    base = 0.5 + (prediction_accuracy * 0.4)

    if recent_failures > 0
        base *= max(0.3, 1.0 - (recent_failures * 0.1))
    end

    if current_state == SYSTEM_STATE_LEARNING
        base *= 0.85
    elseif current_state == SYSTEM_STATE_UNCERTAIN
        base *= 0.7
    elseif current_state == SYSTEM_STATE_PROCESSING
        base *= 1.05
    end

    clamp(base, 0.1, 0.95)
end

function update_capabilities!(self_model::SelfModelType, topic, comprehension)
    topic_str = string(topic)

    capability_updates = Dict{Symbol,Bool}(
        :reasoning => true,
        :conversation => true
    )

    emotional = get(comprehension, :emotional_tone, "")
    if emotional in ["concerned", "warm", "empathetic"]
        capability_updates[:empathy] = true
    end

    prediction_confidence = get(comprehension, :prediction_confidence, 0.5)
    if prediction_confidence > 0.6
        capability_updates[:prediction] = true
    end

    creativity_score = get(comprehension, :creativity_score, 0.0)
    if creativity_score > 0.5
        capability_updates[:creativity] = true
    end

    for (cap, active) in capability_updates
        if active && string(cap) ∉ self_model.capabilities
            push!(self_model.capabilities, string(cap))
        end
    end

    nothing
end

function assess_capability(self_model::SelfModelType, capability::Symbol)::Bool
    string(capability) in self_model.capabilities
end

function assess_capability(self_model::SelfModelType, capability::String)::Bool
    capability in self_model.capabilities
end

function get_capability_confidence(capability::Symbol)::Float64
    get(CAPABILITY_CONFIDENCE, capability, 0.5)
end

function get_uncertainty_level(self_model::SelfModelType)::Float64
    uncertainty = 1.0 - self_model.confidence

    if self_model.current_state == SYSTEM_STATE_LEARNING
        uncertainty = max(uncertainty, 0.4)
    elseif self_model.current_state == SYSTEM_STATE_UNCERTAIN
        uncertainty = max(uncertainty, 0.6)
    elseif length(self_model.limitations) > 3
        uncertainty = max(uncertainty, 0.3)
    end

    uncertainty
end

function know_self_limits(self_model::SelfModelType, task_type::Symbol)::Tuple{Bool,Float64}
    cap_confidence = get_capability_confidence(task_type)
    has_capability = assess_capability(self_model, task_type)
    uncertainty = get_uncertainty_level(self_model)

    can_do = has_capability && (cap_confidence * (1.0 - uncertainty)) > 0.4
    confidence = has_capability ? cap_confidence * (1.0 - uncertainty) : 0.0

    can_do, confidence
end

function get_self_description(self_model::SelfModelType)::Dict{String,Any}
    state_names = Dict(
        SYSTEM_STATE_IDLE => "idle",
        SYSTEM_STATE_PROCESSING => "processing",
        SYSTEM_STATE_CURIOUS => "curious",
        SYSTEM_STATE_UNCERTAIN => "uncertain",
        SYSTEM_STATE_LEARNING => "learning",
        SYSTEM_STATE_STAYING_PRESENT => "staying_present"
    )

    confidence_label = self_model.confidence > 0.8 ? "high" :
                        self_model.confidence > 0.5 ? "medium" : "low"

    uncertainty_level = get_uncertainty_level(self_model)

    Dict{String,Any}(
        "identity" => self_model.identity,
        "current_state" => state_names[self_model.current_state],
        "confidence" => self_model.confidence,
        "confidence_label" => confidence_label,
        "uncertainty_level" => uncertainty_level,
        "capabilities" => self_model.capabilities,
        "limitations" => self_model.limitations
    )
end

function update_confidence!(self_model::SelfModelType, delta::Float64)
    self_model.confidence = clamp(self_model.confidence + delta, 0.1, 0.95)
    nothing
end

function note_failure!(self_model::SelfModelType, failure_type::String)
    if failure_type ∉ self_model.limitations
        push!(self_model.limitations, failure_type)
    end
    self_model.confidence = max(0.1, self_model.confidence - 0.05)
    nothing
end

function note_success!(self_model::SelfModelType, success_type::Symbol)
    cap_str = string(success_type)
    if cap_str ∉ self_model.capabilities
        push!(self_model.capabilities, cap_str)
    end
    self_model.confidence = min(0.95, self_model.confidence + 0.02)
    nothing
end

end # module