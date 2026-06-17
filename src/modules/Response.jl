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

export formulate, adjust_tone, get_onboarding_response, get_persona_intro

const FIRST_TIME_GREETINGS = [
    "Hey there. I'm IngExuity. I'm not just another AI — I learn who you are over time. Right now I'm blank, but that changes with every conversation we have. What's on your mind?",
    "Hi. I'm IngExuity. I become irreplaceable through use, not training. The more we talk, the better I understand you. Shall we start?",
    "Hello. I'm IngExuity. Think of me as a companion who pays attention. I'm currently a blank slate, but I'll learn your patterns, your needs, the things you don't say. Ready to begin?",
]

const RETURNING_GREETINGS = [
    "Good to see you again.",
    "Hey. Back for more?",
    "Good to hear from you.",
    "There you are.",
]

const STAY_PRESENT_TEMPLATES = [
    "I'm here. Take your time.",
    "That sounds heavy. You don't have to figure it out right now.",
    "I'm listening.",
    "There's no rush. I'm here.",
    "That sounds hard. I'm here with you.",
    "Take your time — I'm not going anywhere.",
]

const EMPATHETIC_VALIDATION = [
    "That makes sense.",
    "I can see why that would be that way.",
    "Of course you'd feel that.",
    "That sounds really difficult.",
    "Anyone would feel that way in that situation.",
]

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
    (:emotional, :staying_present, false) => STAY_PRESENT_TEMPLATES,

    # Help seeking
    (:help_seeking, :direct, false) => [
        "What would help most right now?",
        "Let's break it down.",
        "Where do you want to start?",
        "What's the core of the problem?",
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
    (:creative, :direct, false) => [
        "Tell me more about the idea.",
        "What's the core of it?",
        "What would success look like?",
    ],
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

    # Onboarding / first time
    (:onboarding, :warm, false) => FIRST_TIME_GREETINGS,
    (:returning, :warm, false) => RETURNING_GREETINGS,
)

function tone_to_enum(tone)
    tone === :direct && return RESPONSE_TONE_DIRECT
    tone === :warm && return RESPONSE_TONE_WARM
    tone === :playful && return RESPONSE_TONE_PLAYFUL
    tone === :curious && return RESPONSE_TONE_CURIOUS
    tone === :minimal && return RESPONSE_TONE_MINIMAL
    tone === :staying_present && return RESPONSE_TONE_STAYING_PRESENT
    tone === :onboarding && return RESPONSE_TONE_WARM
    tone === :returning && return RESPONSE_TONE_WARM
    tone
end

function formulate(predictions, comprehension; tone=:direct, live_context=nothing)
    topic = comprehension[:topic]
    is_question = comprehension[:is_question]

    key = (topic, tone, is_question)
    candidates = get(RESPONSE_TEMPLATES, key, nothing)
    if candidates === nothing
        key = (:general, tone, is_question)
        candidates = get(RESPONSE_TEMPLATES, key, ["I'm here."])
    end

    idx = isempty(predictions) ? 1 : (length(predictions) % length(candidates)) + 1
    content = candidates[idx]

    # Incorporate live data if available
    if live_context !== nothing
        content = _incorporate_live_data(content, live_context, is_question)
    end

    ResponseType(content, tone_to_enum(tone), 0.8, 0.7, false)
end

function _incorporate_live_data(content::String, live_context::Dict{String,Any}, is_question::Bool)::String
    context = get(live_context, "context", "")
    if isempty(context)
        return content
    end

    query_type = get(live_context, "query_type", "general")
    results = get(live_context, "results", [])

    if is_question && !isempty(results)
        # For questions, provide a direct answer from live data
        top_result = results[1]
        title = get(top_result, "title", "")
        snippet = get(top_result, "snippet", "")

        if query_type == "weather"
            return "Based on current conditions: $snippet"
        elseif query_type == "stock"
            return "Here's the latest on that: $snippet"
        elseif query_type == "news"
            return "Current news: $snippet"
        elseif query_type == "factual"
            return "$snippet (Source: $title)"
        else
            return "$snippet"
        end
    end

    # For statements, acknowledge with live context
    return content
end

function get_onboarding_response(turn_count::Int64)::ResponseType
    if turn_count == 0
        idx = rand(1:length(FIRST_TIME_GREETINGS))
        ResponseType(FIRST_TIME_GREETINGS[idx], RESPONSE_TONE_WARM, 0.9, 0.5, false)
    else
        idx = rand(1:length(RETURNING_GREETINGS))
        ResponseType(RETURNING_GREETINGS[idx], RESPONSE_TONE_WARM, 0.9, 0.8, false)
    end
end

function get_persona_intro()::String
    """
    I'm IngExuity.

    I don't just answer questions — I learn who you are. Your patterns, your stress signals, the things you don't say out loud.

    The more we talk, the better I anticipate what you need. And when you're having a hard time, I stay present. I don't rush to fix things.

    Right now I'm a blank slate. But that changes with every conversation.

    Ready to start?
    """
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

function get_stay_present_response(internal)::String
    idx = rand(1:length(STAY_PRESENT_TEMPLATES))
    STAY_PRESENT_TEMPLATES[idx]
end

function get_empathetic_validation()::String
    idx = rand(1:length(EMPATHETIC_VALIDATION))
    EMPATHETIC_VALIDATION[idx]
end

end # module Response
