# ============================================================================
# Response.jl — Shape and formulate the response
# No external ML deps — template-based, works on Railway CPU
# Cold start: rich templates so IngExuity isn't blank on day 1
# ============================================================================
module Response

using ..Types: Response as ResponseType, ResponseTone as ResponseToneType,
               RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
               RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS,
               RESPONSE_TONE_MINIMAL, RESPONSE_TONE_STAYING_PRESENT

export formulate, adjust_tone

# Template responses keyed by topic × tone × is_question
# IngExuity activates immediately — not blank on day 1
const RESPONSE_TEMPLATES = Dict(
    # General
    (:general, :direct, false) => [
        "Go on.",
        "I'm here.",
        "What else is on your mind?",
        "Tell me more about that.",
        "And?",
    ],
    (:general, :direct, true) => [
        "That's a good question.",
        "What do you think?",
        "I'm curious what brought you to that.",
    ],
    (:general, :warm, false) => [
        "I'm glad you shared that.",
        "That sounds meaningful.",
        "I'm here with you.",
    ],
    (:general, :warm, true) => [
        "That's worth thinking about.",
        "I'm glad you asked.",
    ],
    (:general, :curious, false) => [
        "That's interesting.",
        "I'm curious about your perspective.",
        "What do you make of that?",
    ],
    (:general, :curious, true) => [
        "What would an ideal version of that look like?",
        "How do you think it works?",
    ],
    (:general, :playful, false) => [
        "That's a fun one.",
        "I like where this is going.",
    ],
    (:general, :minimal, false) => ["Mm.", "Right.", "And?"],

    # Emotional / staying present
    (:emotional, :direct, false) => [
        "That's hard. I'm here.",
        "That sounds really difficult.",
        "There's a lot in that.",
    ],
    (:emotional, :staying_present, false) => [
        "I'm here. Take your time.",
        "That sounds heavy. You don't have to figure it out right now.",
        "I'm listening.",
    ],

    # Help seeking
    (:help_seeking, :direct, false) => [
        "What would help most right now?",
        "Let's break it down.",
        "Where do you want to start?",
    ],
    (:help_seeking, :direct, true) => [
        "Let's figure this out together.",
        "What have you tried so far?",
    ],

    # Positive
    (:positive, :warm, false) => [
        "That's great to hear.",
        "I love that for you.",
        "What's making it so good?",
    ],

    # Creative
    (:creative, :curious, false) => [
        "What inspired that?",
        "I like that direction.",
        "What else is connected to this?",
    ],
    (:creative, :curious, true) => [
        "Let's explore that.",
        "What would it look like fully realized?",
    ],

    # Work
    (:work, :direct, false) => [
        "How's that looking?",
        "What's the latest on that?",
        "What matters most about it?",
    ],
    (:work, :direct, true) => [
        "What are the constraints?",
        "What would help you move forward?",
    ],
    (:work, :curious, true) => [
        "Have you considered it from that angle?",
        "What does success look like?",
    ],

    # Family
    (:family, :warm, false) => [
        "Family matters a lot.",
        "How are you navigating that?",
        "What role do you want to play in it?",
    ],
)

function tone_to_enum(tone)
    tone === :direct && return RESPONSE_TONE_DIRECT
    tone === :warm && return RESPONSE_TONE_WARM
    tone === :playful && return RESPONSE_TONE_PLAYFUL
    tone === :curious && return RESPONSE_TONE_CURIOUS
    tone === :minimal && return RESPONSE_TONE_MINIMAL
    tone === :staying_present && return RESPONSE_TONE_STAYING_PRESENT
    # Also accept ResponseTone enum values directly
    tone
end

function formulate(predictions, comprehension; tone=:direct)
    topic = comprehension[:topic]
    is_question = comprehension[:is_question]

    # Look up template
    key = (topic, tone, is_question)
    candidates = get(RESPONSE_TEMPLATES, key, nothing)
    if candidates === nothing
        # Fallback: general topic
        key = (:general, tone, is_question)
        candidates = get(RESPONSE_TEMPLATES, key, ["I'm here."])
    end

    # Pick deterministically but variedly based on prediction count
    idx = isempty(predictions) ? 1 : (length(predictions) % length(candidates)) + 1
    content = candidates[idx]

    ResponseType(content, tone_to_enum(tone), 0.8, 0.7, false)
end

function adjust_tone(response::ResponseType, internal)
    if internal.stress_level > 0.5
        ResponseType(response.content, RESPONSE_TONE_DIRECT, 0.9, response.confidence, response.should_retry)
    elseif internal.affective_state == "warm"
        ResponseType(response.content, RESPONSE_TONE_WARM, 0.85, response.confidence, response.should_retry)
    else
        response
    end
end

end # module Response
